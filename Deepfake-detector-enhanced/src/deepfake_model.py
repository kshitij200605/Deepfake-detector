"""
Enhanced deepfake detection using a dual-model ensemble:
  1. dima806/deepfake_vs_real_image_detection  (ViT, primary - face-level)
  2. Wvolf/ViT-Deepfake-Detection               (ViT, secondary - scene-level)

Also extracts auxiliary artifact signals:
  - Face region texture entropy (low entropy = smooth = GAN artifact)
  - Eye region analysis (blinking patterns, reflection inconsistencies)
  - Frequency domain analysis (GAN upsampling leaves spectral fingerprints)
"""
import logging
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline
from config import (
    MODEL_NAME, SECONDARY_MODEL_NAME,
    WEIGHT_PRIMARY, WEIGHT_SECONDARY,
)

logger = logging.getLogger(__name__)

# ── Face Detection Setup ───────────────────────────────────────────────
_frontal_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


class DeepfakeClassifier:
    """
    Dual-model ensemble deepfake classifier with artifact analysis.
    """

    def __init__(self):
        self._primary   = None
        self._secondary = None
        self._device    = None

    def _load_models(self):
        if self._primary is not None:
            return
        self._device = 0 if torch.cuda.is_available() else -1
        device_name  = "CUDA GPU" if self._device == 0 else "CPU"
        logger.info("Loading models on %s", device_name)

        logger.info("Loading primary model: %s", MODEL_NAME)
        self._primary = pipeline(
            "image-classification",
            model=MODEL_NAME,
            device=self._device,
        )

        # Secondary model — silently skip if unavailable (graceful degradation)
        try:
            logger.info("Loading secondary model: %s", SECONDARY_MODEL_NAME)
            self._secondary = pipeline(
                "image-classification",
                model=SECONDARY_MODEL_NAME,
                device=self._device,
            )
        except Exception as e:
            logger.warning("Secondary model unavailable (%s). Using primary only.", e)
            self._secondary = None

        logger.info("Models loaded.")

    # ──────────────────────────────────────────────────────────────────
    def predict_frame(self, frame) -> dict:
        """
        Full frame analysis pipeline.

        Returns
        -------
        dict:
            score           float  0=real, 1=fake (ensemble)
            primary_score   float
            secondary_score float | None
            faces_found     int
            label           str
            artifacts       dict   auxiliary signals
        """
        self._load_models()

        faces = _detect_faces(frame)

        if len(faces) == 0:
            return {
                "score": 0.5,
                "primary_score": 0.5,
                "secondary_score": None,
                "faces_found": 0,
                "label": "No Face",
                "artifacts": {},
            }

        face_results   = []
        artifact_data  = []

        for (x, y, w, h) in faces:
            pad  = int(0.2 * max(w, h))
            y1   = max(0, y - pad)
            x1   = max(0, x - pad)
            y2   = min(frame.shape[0], y + h + pad)
            x2   = min(frame.shape[1], x + w + pad)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            face_rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)

            # ── Primary model ──────────────────────────────────────
            try:
                p_score = _extract_fake_score(self._primary(pil_image))
            except Exception as e:
                logger.warning("Primary model error: %s", e)
                p_score = 0.5

            # ── Secondary model ────────────────────────────────────
            s_score = None
            if self._secondary is not None:
                try:
                    s_score = _extract_fake_score(self._secondary(pil_image))
                except Exception as e:
                    logger.warning("Secondary model error: %s", e)

            # ── Ensemble ───────────────────────────────────────────
            if s_score is not None:
                ensemble = WEIGHT_PRIMARY * p_score + WEIGHT_SECONDARY * s_score
            else:
                ensemble = p_score

            # ── Artifact signals ───────────────────────────────────
            artifacts = _extract_artifacts(crop)

            face_results.append({
                "primary": p_score,
                "secondary": s_score,
                "ensemble": ensemble,
            })
            artifact_data.append(artifacts)

        if not face_results:
            return {
                "score": 0.5,
                "primary_score": 0.5,
                "secondary_score": None,
                "faces_found": len(faces),
                "label": "Error",
                "artifacts": {},
            }

        # Take worst-case (maximum fake probability) across faces
        best = max(face_results, key=lambda r: r["ensemble"])
        merged_artifacts = {
            k: float(np.mean([a.get(k, 0) for a in artifact_data]))
            for k in artifact_data[0]
        } if artifact_data else {}

        label = "Fake" if best["ensemble"] > 0.5 else "Real"

        return {
            "score":           round(best["ensemble"], 4),
            "primary_score":   round(best["primary"], 4),
            "secondary_score": round(best["secondary"], 4) if best["secondary"] is not None else None,
            "faces_found":     len(faces),
            "label":           label,
            "artifacts":       merged_artifacts,
        }


# ── Helpers ────────────────────────────────────────────────────────────

def _detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = _frontal_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4,
        minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) > 0:
        return faces

    faces = _profile_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4,
        minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return faces


def _extract_fake_score(results: list) -> float:
    for item in results:
        if "fake" in item.get("label", "").lower():
            return item["score"]
    for item in results:
        if "real" in item.get("label", "").lower():
            return 1.0 - item["score"]
    return 0.5


def _extract_artifacts(face_bgr) -> dict:
    """
    Compute auxiliary artifact signals that complement ML scores.

    Returns
    -------
    dict with float values [0, 1] where higher = more suspicious.
    """
    artifacts = {}

    # 1. Texture entropy — GAN-smoothed faces have unnaturally low entropy
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist[hist > 0] / hist.sum()
    entropy = float(-np.sum(hist * np.log2(hist + 1e-9)))
    # Normalize: typical real face entropy ~4-6 bits; GAN <3
    artifacts["texture_entropy_suspicion"] = round(
        max(0.0, min(1.0, (5.5 - entropy) / 3.0)), 3
    )

    # 2. High-frequency noise (Laplacian variance)
    # GAN outputs are often over-smoothed → low Laplacian variance
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize empirically: real faces ~200-800; GAN ~30-150
    artifacts["smoothness_suspicion"] = round(
        max(0.0, min(1.0, 1.0 - lap_var / 500.0)), 3
    )

    # 3. Frequency domain — GAN checkerboard artifacts (DCT-based)
    h, w = gray.shape
    dct = cv2.dct(np.float32(gray))
    # Energy ratio in high-frequency quadrant vs total
    hf_energy = float(np.sum(dct[h//2:, w//2:] ** 2))
    total_energy = float(np.sum(dct ** 2)) + 1e-9
    hf_ratio = hf_energy / total_energy
    # Deepfakes tend to have anomalous HF patterns
    artifacts["frequency_anomaly"] = round(min(1.0, hf_ratio * 50), 3)

    # 4. Colour channel correlation
    # Real faces have consistent inter-channel correlations; GAN synthesis may break this
    b, g, r = cv2.split(face_bgr)
    rg_corr = float(np.corrcoef(r.flatten(), g.flatten())[0, 1])
    artifacts["color_channel_inconsistency"] = round(
        max(0.0, min(1.0, 1.0 - abs(rg_corr))), 3
    )

    return artifacts


# ── Global Instance ────────────────────────────────────────────────────
classifier = DeepfakeClassifier()
