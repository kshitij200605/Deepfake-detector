"""
Enhanced video analysis pipeline for deepfake detection.

Improvements over original:
  1. Dual-model ensemble scoring
  2. Temporal consistency analysis (frame-to-frame deltas)
  3. Artifact signal aggregation (texture, frequency, colour)
  4. Per-segment scoring (breaks video into thirds for localisation)
  5. Key-frame thumbnail extraction for report
  6. Richer metadata returned (FPS, codec, resolution)
  7. Adaptive sampling with hard frame cap for long videos
"""
import logging
import base64
import io
import cv2
import numpy as np
from PIL import Image

from deepfake_model import classifier
from config import (
    FRAMES_PER_SECOND,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    TEMPORAL_WINDOW,
    TEMPORAL_VARIANCE_THRESHOLD,
    MAX_FRAMES_TO_ANALYZE,
    THUMBNAIL_COUNT,
)

logger = logging.getLogger(__name__)


def analyze_video(video_path: str, progress_callback=None) -> dict:
    """
    Full video analysis pipeline.

    Returns an enriched result dict including:
      - verdict / confidence / confidence_level
      - primary_model_score, secondary_model_score
      - ensemble_score
      - temporal_consistency_score (0=erratic, 1=consistent)
      - artifact_signals  (aggregated auxiliary metrics)
      - segment_scores    (early / mid / late video thirds)
      - frame_scores      list[float]
      - key_frames        list[base64-PNG strings]  (for UI thumbnails)
      - video_meta        (fps, resolution, codec, duration, total_frames)
      - frames_analyzed, faces_detected, total_frames, video_duration
      - warnings          list[str]
    """
    warnings = []

    # ── Open video ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _error_result("Cannot open video file")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps if fps > 0 else 0.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int   = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec        = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()

    logger.info(
        "Video: %dx%d @ %.1f FPS, %d frames (%.1fs), codec=%s",
        width, height, fps, total_frames, duration, codec,
    )

    # ── Adaptive frame sampling ─────────────────────────────────────────
    frame_skip = max(1, int(fps / FRAMES_PER_SECOND))
    raw_expected = max(1, total_frames // frame_skip)
    if raw_expected > MAX_FRAMES_TO_ANALYZE:
        frame_skip = max(1, total_frames // MAX_FRAMES_TO_ANALYZE)
    expected_samples = max(1, total_frames // frame_skip)

    if progress_callback:
        progress_callback(2, "Initialising detection models…")

    # ── Main frame loop ─────────────────────────────────────────────────
    frame_scores        = []   # ensemble fake-probability per frame
    primary_scores      = []
    secondary_scores    = []
    artifact_history    = []
    total_faces         = 0
    frames_with_no_face = 0
    frame_number        = 0
    analyzed_count      = 0

    # Key-frame extraction
    key_frame_interval  = max(1, expected_samples // THUMBNAIL_COUNT)
    key_frames_b64      = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        if frame_number % frame_skip != 0:
            continue

        result = classifier.predict_frame(frame)

        if result["faces_found"] == 0:
            frames_with_no_face += 1
        else:
            frame_scores.append(result["score"])
            primary_scores.append(result["primary_score"])
            if result["secondary_score"] is not None:
                secondary_scores.append(result["secondary_score"])
            total_faces += result["faces_found"]
            if result["artifacts"]:
                artifact_history.append(result["artifacts"])

        analyzed_count += 1

        # Save key frames (thumbnail)
        if analyzed_count % key_frame_interval == 0 and len(key_frames_b64) < THUMBNAIL_COUNT:
            b64 = _frame_to_b64(frame)
            if b64:
                key_frames_b64.append({
                    "b64": b64,
                    "frame_no": frame_number,
                    "score": result["score"],
                    "label": result["label"],
                })

        if progress_callback:
            pct = min(97, 5 + int((analyzed_count / expected_samples) * 88))
            progress_callback(pct, f"Frame {analyzed_count}/{expected_samples} · {result['label']}")

    cap.release()

    # ── Edge cases ──────────────────────────────────────────────────────
    if analyzed_count == 0:
        return _error_result("No frames could be read from the video")

    if not frame_scores:
        warnings.append("No faces detected in any frame — unable to perform deepfake analysis")
        return {
            "verdict": "INCONCLUSIVE",
            "confidence": 0.0,
            "confidence_level": "LOW",
            "ensemble_score": 0.5,
            "primary_model_score": None,
            "secondary_model_score": None,
            "temporal_consistency_score": None,
            "artifact_signals": {},
            "segment_scores": {},
            "frames_analyzed": analyzed_count,
            "faces_detected": 0,
            "total_frames": total_frames,
            "video_duration": round(duration, 1),
            "frame_scores": [],
            "key_frames": key_frames_b64,
            "video_meta": _meta(fps, width, height, codec, duration, total_frames),
            "warnings": warnings,
        }

    if frames_with_no_face > analyzed_count * 0.6:
        warnings.append(
            f"Faces absent in {frames_with_no_face}/{analyzed_count} sampled frames — "
            "results may be less reliable"
        )

    # ── Aggregate scores ────────────────────────────────────────────────
    avg_score        = float(np.mean(frame_scores))
    avg_primary      = float(np.mean(primary_scores)) if primary_scores else None
    avg_secondary    = float(np.mean(secondary_scores)) if secondary_scores else None

    # ── Temporal consistency ────────────────────────────────────────────
    temporal_consistency = _temporal_consistency(frame_scores, TEMPORAL_WINDOW)
    score_variance = float(np.var(frame_scores)) if len(frame_scores) > 1 else 0.0

    if score_variance > TEMPORAL_VARIANCE_THRESHOLD:
        warnings.append(
            "High score variance across frames — detection may be less reliable"
        )
    if temporal_consistency < 0.5:
        warnings.append(
            "Inconsistent detection across frames — possible partial manipulation"
        )

    # ── Artifact signals ────────────────────────────────────────────────
    artifact_signals = {}
    if artifact_history:
        for key in artifact_history[0]:
            values = [a[key] for a in artifact_history if key in a]
            artifact_signals[key] = round(float(np.mean(values)), 3)

    # ── Segment scores (early / mid / late) ────────────────────────────
    segment_scores = _compute_segment_scores(frame_scores)

    # ── Verdict ─────────────────────────────────────────────────────────
    verdict, confidence_level = _determine_verdict(avg_score)
    confidence = round(abs(avg_score - 0.5) * 2, 3)

    if progress_callback:
        progress_callback(100, "Analysis complete")

    return {
        "verdict":                 verdict,
        "confidence":              confidence,
        "confidence_level":        confidence_level,
        "ensemble_score":          round(avg_score, 4),
        "primary_model_score":     round(avg_primary, 4) if avg_primary is not None else None,
        "secondary_model_score":   round(avg_secondary, 4) if avg_secondary is not None else None,
        "temporal_consistency_score": round(temporal_consistency, 3),
        "artifact_signals":        artifact_signals,
        "segment_scores":          segment_scores,
        "frames_analyzed":         analyzed_count,
        "faces_detected":          total_faces,
        "total_frames":            total_frames,
        "video_duration":          round(duration, 1),
        "frame_scores":            [round(s, 3) for s in frame_scores],
        "key_frames":              key_frames_b64,
        "video_meta":              _meta(fps, width, height, codec, duration, total_frames),
        "warnings":                warnings,
    }


# ── Helpers ────────────────────────────────────────────────────────────

def _temporal_consistency(scores: list, window: int) -> float:
    """
    Returns 0 (erratic) to 1 (perfectly consistent).
    Uses rolling-window standard deviation.
    """
    if len(scores) < 2:
        return 1.0
    deviations = []
    for i in range(len(scores) - window + 1):
        chunk = scores[i:i + window]
        deviations.append(float(np.std(chunk)))
    avg_dev = float(np.mean(deviations)) if deviations else 0.0
    return round(max(0.0, 1.0 - avg_dev * 4), 3)


def _compute_segment_scores(scores: list) -> dict:
    if len(scores) < 3:
        avg = float(np.mean(scores)) if scores else 0.5
        return {"early": avg, "mid": avg, "late": avg}
    n   = len(scores)
    t   = n // 3
    return {
        "early": round(float(np.mean(scores[:t])), 3),
        "mid":   round(float(np.mean(scores[t:2*t])), 3),
        "late":  round(float(np.mean(scores[2*t:])), 3),
    }


def _determine_verdict(score: float):
    if score >= CONFIDENCE_HIGH:
        return "FAKE", "HIGH"
    if score >= CONFIDENCE_MEDIUM:
        return "FAKE", "MEDIUM"
    if score <= (1 - CONFIDENCE_HIGH):
        return "REAL", "HIGH"
    if score <= (1 - CONFIDENCE_MEDIUM):
        return "REAL", "MEDIUM"
    return "INCONCLUSIVE", "LOW"


def _frame_to_b64(frame, max_dim: int = 320) -> str | None:
    try:
        h, w = frame.shape[:2]
        scale = min(max_dim / w, max_dim / h, 1.0)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        pil   = Image.fromarray(rgb)
        buf   = io.BytesIO()
        pil.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        logger.warning("Key-frame encode failed: %s", e)
        return None


def _meta(fps, w, h, codec, dur, total):
    return {
        "fps":        round(fps, 1),
        "resolution": f"{w}×{h}",
        "codec":      codec,
        "duration":   round(dur, 1),
        "total_frames": total,
    }


def _error_result(message: str) -> dict:
    return {
        "verdict":          "ERROR",
        "confidence":       0.0,
        "confidence_level": "LOW",
        "ensemble_score":   0.5,
        "primary_model_score": None,
        "secondary_model_score": None,
        "temporal_consistency_score": None,
        "artifact_signals": {},
        "segment_scores":   {},
        "frames_analyzed":  0,
        "faces_detected":   0,
        "total_frames":     0,
        "video_duration":   0.0,
        "frame_scores":     [],
        "key_frames":       [],
        "video_meta":       {},
        "warnings":         [message],
    }
