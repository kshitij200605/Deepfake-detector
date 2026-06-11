# DeepGuard v2 — Enhanced Deepfake Detector

> **Enterprise-grade AI video forensics.** Detect AI-generated and deepfake video with a dual-model ensemble, temporal analysis, artifact forensics, and a polished dark-mode UI.

---

## What's New in v2 (vs Original)

| Feature | Original | v2 Enhanced |
|---|---|---|
| Detection models | 1 (primary ViT) | **2 (dual-model ensemble)** |
| Artifact analysis | None | **4 artifact signals** |
| Temporal consistency | Basic variance check | **Rolling-window consistency score** |
| Segment analysis | None | **Early / Mid / Late scoring** |
| Key frame thumbnails | None | **Up to 6 annotated key frames** |
| Score timeline chart | None | **Canvas-rendered frame-by-frame chart** |
| Video metadata | None | **FPS, codec, resolution, duration** |
| Report export | None | **JSON forensic report download** |
| Upload limit | 100 MB | **200 MB** |
| Frame sampling rate | 2 fps | **3 fps (capped at 120 frames)** |
| UI theme | Light glassmorphism | **Dark enterprise + animated grid** |
| Progress indicators | Single spinner | **6-step labelled pipeline** |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/kshitij200605/Deepfake-detector.git
cd Deepfake-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
cd src
python app.py
# Open http://localhost:5000
```

Models are downloaded from HuggingFace on first run (~700 MB, cached to `~/.cache/huggingface`).

---

## Architecture

```
DeepGuard v2
├── src/
│   ├── app.py              Flask app + /api/export endpoint
│   ├── detector.py         Video pipeline (sampling, temporal, segments)
│   ├── deepfake_model.py   Dual-model ensemble + artifact extraction
│   ├── malware_scanner.py  Magic-byte + executable signature scan
│   └── config.py           All tuneable constants
├── templates/
│   └── index.html          Full dark-mode enterprise UI
└── requirements.txt
```

---

## Detection Pipeline

```
Upload → Malware Scan → Frame Sampling
     → [Primary ViT] + [Secondary ViT] → Ensemble Score
     → Face Extraction → Artifact Signals
     → Temporal Consistency Analysis
     → Segment Scoring (early/mid/late)
     → Verdict + Confidence
```

### Models Used
- **Primary:** `dima806/deepfake_vs_real_image_detection` (weight: 65%)
- **Secondary:** `Wvolf/ViT-Deepfake-Detection` (weight: 35%, graceful degradation if unavailable)

### Artifact Signals (4)
| Signal | Description |
|---|---|
| Texture Entropy Suspicion | GAN-synthesised faces are often unnaturally smooth (low entropy) |
| Over-smoothness | Laplacian variance — over-smoothed regions indicate upsampling artifacts |
| Frequency Anomaly | DCT high-frequency energy ratio — GAN checkerboard/upsampling artifacts |
| Colour Inconsistency | Inter-channel correlation breakdown — real faces have coherent RGB channels |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve UI |
| `POST` | `/api/analyze` | Upload video, start async analysis |
| `GET` | `/api/status/<task_id>` | Poll progress + results |
| `GET` | `/api/export/<task_id>` | Download JSON forensic report |
| `GET` | `/api/export/<task_id>?include_frames=1` | Report with base64 key frames |

### Result Schema (`/api/status`)
```json
{
  "verdict":                  "FAKE | REAL | INCONCLUSIVE",
  "confidence":               0.87,
  "confidence_level":         "HIGH | MEDIUM | LOW",
  "ensemble_score":           0.93,
  "primary_model_score":      0.91,
  "secondary_model_score":    0.96,
  "temporal_consistency_score": 0.78,
  "artifact_signals": {
    "texture_entropy_suspicion":   0.62,
    "smoothness_suspicion":        0.48,
    "frequency_anomaly":           0.31,
    "color_channel_inconsistency": 0.22
  },
  "segment_scores": { "early": 0.88, "mid": 0.95, "late": 0.90 },
  "frames_analyzed":  64,
  "faces_detected":   128,
  "total_frames":     900,
  "video_duration":   30.0,
  "key_frames":       [...],
  "video_meta": { "fps": 30, "resolution": "1920×1080", "codec": "avc1", "duration": 30.0 },
  "warnings":         ["High score variance across frames"]
}
```

---

## Configuration (`src/config.py`)

```python
FRAMES_PER_SECOND    = 3      # Sampling rate
MAX_FRAMES_TO_ANALYZE= 120    # Hard cap for long videos
CONFIDENCE_HIGH      = 0.72   # Threshold for HIGH verdict
CONFIDENCE_MEDIUM    = 0.55   # Threshold for MEDIUM verdict
WEIGHT_PRIMARY       = 0.65   # Primary model weight
WEIGHT_SECONDARY     = 0.35   # Secondary model weight
TEMPORAL_WINDOW      = 5      # Frames for consistency rolling window
```
