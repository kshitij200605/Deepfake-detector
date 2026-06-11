# 🛡️ Deepfake Detector

AI-powered tool that detects deepfake videos to help combat cybercrime, misinformation, and identity fraud.

Upload a video → the system extracts faces → a Vision Transformer classifies each face as **Real** or **Fake** → you get a clear verdict with confidence scoring.

---

## Why This Matters

Deepfake technology is increasingly being weaponized for:
- **Identity fraud** — impersonating individuals in video calls
- **Financial scams** — fake CEO videos authorizing fraudulent transfers
- **Misinformation** — fabricated political or news footage
- **Harassment** — non-consensual synthetic media

This tool provides an accessible way to verify video authenticity.

---

## Features

- 🎯 **Real deepfake detection** — Uses a ViT model fine-tuned on deepfake datasets (not generic ImageNet)
- 🔒 **Malware scanning** — Uploaded videos are scanned for embedded executables and signature spoofing
- ⚡ **Async processing** — Non-blocking analysis with real-time progress updates
- 📊 **Detailed results** — Confidence scoring, frame-by-frame analysis, temporal consistency checks
- 🧹 **Auto cleanup** — Uploaded files are deleted after analysis
- 🖥️ **Modern UI** — Clean drag-and-drop interface with live progress and visual confidence gauges

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| AI Model | HuggingFace Transformers (ViT) |
| Face Detection | OpenCV Haar Cascades |
| Frontend | HTML5, CSS3, Vanilla JS |
| Async | Python threading + polling API |

---

## Project Structure

```
deepfake-detector/
├── src/
│   ├── app.py              # Flask app with API endpoints
│   ├── config.py           # Centralized configuration
│   ├── deepfake_model.py   # HuggingFace deepfake classifier
│   ├── detector.py         # Video analysis pipeline
│   └── malware_scanner.py  # Upload security scanning
├── templates/
│   └── index.html          # Frontend UI
├── uploads/                # Temp storage (auto-created, auto-cleaned)
├── requirements.txt
└── README.md
```

---

## Setup & Run

```bash
# Clone the repository
git clone https://github.com/kshitij200605/Deepfake-detector.git
cd deepfake-detector

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
python src/app.py
```

The app will be available at **http://localhost:5000**

> **Note**: On first run, the AI model (~350MB) will be downloaded from HuggingFace. This only happens once.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main UI |
| `POST` | `/api/analyze` | Upload video for analysis (multipart form) |
| `GET` | `/api/status/<task_id>` | Poll analysis progress |

---

## How It Works

1. **Upload** — Video file is uploaded and validated
2. **Security Scan** — File is checked for malware indicators (magic bytes, embedded executables)
3. **Frame Sampling** — Frames are extracted at ~2 per second (adaptive to video FPS)
4. **Face Detection** — OpenCV detects faces in each sampled frame
5. **Classification** — Each face is classified by a Vision Transformer trained on deepfake datasets
6. **Aggregation** — Per-frame scores are aggregated with temporal consistency analysis
7. **Verdict** — Final result: REAL, FAKE, or INCONCLUSIVE with confidence level

---

## Model

Uses [`dima806/deepfake_vs_real_image_detection`](https://huggingface.co/dima806/deepfake_vs_real_image_detection) — a Vision Transformer (ViT) fine-tuned specifically for distinguishing deepfake faces from real ones.

---

## License

MIT
