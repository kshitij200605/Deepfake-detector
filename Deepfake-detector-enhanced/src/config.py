"""
Centralized configuration for the Enhanced Deepfake Detector application.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

# ── Upload Limits ──────────────────────────────────────────────────────
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB (increased)

ALLOWED_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".3gp"
}

# ── Models ─────────────────────────────────────────────────────────────
# Primary deepfake detection model (ViT-based, face-level)
MODEL_NAME = "dima806/deepfake_vs_real_image_detection"

# Secondary scene/frame-level model for cross-validation
SECONDARY_MODEL_NAME = "Wvolf/ViT-Deepfake-Detection"

# ── Video Analysis ─────────────────────────────────────────────────────
FRAMES_PER_SECOND = 3          # Increased from 2 for better coverage
MAX_FRAMES_TO_ANALYZE = 120    # Cap for very long videos
THUMBNAIL_COUNT = 6            # Number of key frames to extract for report

# Confidence thresholds for verdict levels
CONFIDENCE_HIGH   = 0.72
CONFIDENCE_MEDIUM = 0.55

# ── Multi-Model Ensemble Weights ───────────────────────────────────────
WEIGHT_PRIMARY    = 0.65
WEIGHT_SECONDARY  = 0.35

# ── Temporal Analysis ──────────────────────────────────────────────────
# Number of consecutive frames to consider for temporal consistency
TEMPORAL_WINDOW = 5
TEMPORAL_VARIANCE_THRESHOLD = 0.12  # Tightened

# ── Artifact Detection ─────────────────────────────────────────────────
# JPEG compression ratio (too low = over-compressed = possible sign)
COMPRESSION_SUSPICION_RATIO = 0.3

# ── Malware Scanning ──────────────────────────────────────────────────
VIDEO_SIGNATURES = {
    b"ftyp":         "mp4/mov",
    b"RIFF":         "avi",
    b"\x1a\x45\xdf\xa3": "mkv/webm",
    b"FLV":          "flv",
    b"\x30\x26\xb2\x75": "wmv",
    b"ftyp3g":       "3gp",
}

EXECUTABLE_SIGNATURES = [
    b"\x7fELF",
    b"PK\x03\x04",
    b"\xca\xfe\xba\xbe",
    b"<script",
    b"powershell",
    b"cmd.exe",
]

SCAN_CHUNK_SIZE  = 8192
SCAN_MAX_BYTES   = 5 * 1024 * 1024
