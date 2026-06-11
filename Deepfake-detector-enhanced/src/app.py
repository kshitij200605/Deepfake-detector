"""
Enhanced Flask application for the Deepfake Detector.

New endpoints:
  GET  /                    — Serve the main UI
  POST /api/analyze         — Upload video, run security scan, start async analysis
  GET  /api/status/<id>     — Poll task progress and results
  GET  /api/export/<id>     — Download JSON forensic report for a completed task
"""
import os
import uuid
import json
import logging
import threading
from datetime import datetime, timezone
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, Response
from config import BASE_DIR, UPLOAD_FOLDER, MAX_CONTENT_LENGTH, ALLOWED_EXTENSIONS
from malware_scanner import scan_file
from detector import analyze_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

tasks: dict[str, dict] = {}
tasks_lock = threading.Lock()


def _update_task(task_id: str, **kwargs):
    with tasks_lock:
        if task_id in tasks:
            tasks[task_id].update(kwargs)


# ── Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"success": False, "error": "No file selected"}), 400

    original_name = secure_filename(file.filename)
    if not original_name:
        return jsonify({"success": False, "error": "Invalid filename"}), 400

    ext = os.path.splitext(original_name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return jsonify({
            "success": False,
            "error": f"File type '{ext}' not allowed. Accepted: {allowed}"
        }), 400

    unique_name = f"{uuid.uuid4().hex}{ext}"
    filepath    = os.path.join(UPLOAD_FOLDER, unique_name)

    try:
        file.save(filepath)
    except Exception as e:
        logger.error("Failed to save upload: %s", e)
        return jsonify({"success": False, "error": "Failed to save file"}), 500

    # Malware scan
    scan_result = scan_file(filepath)
    if not scan_result.is_safe:
        _cleanup_file(filepath)
        logger.warning("Malware scan blocked %s: %s", original_name, scan_result.threats)
        return jsonify({
            "success":  False,
            "error":    "File failed security scan",
            "threats":  scan_result.threats,
        }), 400

    task_id = uuid.uuid4().hex[:12]
    with tasks_lock:
        tasks[task_id] = {
            "status":    "processing",
            "progress":  0,
            "message":   "Starting analysis…",
            "result":    None,
            "filename":  original_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    threading.Thread(
        target=_process_video_task,
        args=(task_id, filepath),
        daemon=True,
    ).start()

    logger.info("Task %s started for %s", task_id, original_name)
    return jsonify({"success": True, "task_id": task_id})


@app.route("/api/status/<task_id>")
def api_status(task_id):
    with tasks_lock:
        task = tasks.get(task_id)
    if task is None:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task)


@app.route("/api/export/<task_id>")
def api_export(task_id):
    """
    Download a completed analysis as a JSON forensic report.
    Strips base64 key-frames to keep the export lean (optional param: ?include_frames=1).
    """
    with tasks_lock:
        task = tasks.get(task_id)

    if task is None:
        return jsonify({"error": "Task not found"}), 404
    if task["status"] != "complete":
        return jsonify({"error": "Analysis not complete yet"}), 400

    report = {
        "report_id":     task_id,
        "filename":      task.get("filename", "unknown"),
        "analyzed_at":   task.get("created_at", ""),
        "exported_at":   datetime.now(timezone.utc).isoformat(),
        "tool_version":  "2.0-enhanced",
        "result":        task["result"],
    }

    # Strip large base64 blobs unless caller opts in
    include_frames = request.args.get("include_frames", "0") == "1"
    if not include_frames and report["result"] and "key_frames" in report["result"]:
        report["result"] = {**report["result"], "key_frames": []}

    fname = f"deepfake_report_{task_id}.json"
    return Response(
        json.dumps(report, indent=2),
        mimetype="application/json",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# ── Background Processing ─────────────────────────────────────────────

def _process_video_task(task_id: str, filepath: str):
    try:
        def progress_callback(pct, msg):
            _update_task(task_id, progress=pct, message=msg)

        result = analyze_video(filepath, progress_callback=progress_callback)

        _update_task(
            task_id,
            status="complete",
            progress=100,
            message="Analysis complete",
            result=result,
        )
        logger.info("Task %s: %s (confidence=%.2f)", task_id, result.get("verdict"), result.get("confidence", 0))

    except Exception as e:
        logger.exception("Task %s failed: %s", task_id, e)
        _update_task(
            task_id,
            status="error",
            progress=0,
            message=f"Analysis failed: {e}",
            result=None,
        )
    finally:
        _cleanup_file(filepath)


# ── Utilities ──────────────────────────────────────────────────────────

def _cleanup_file(filepath: str):
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except OSError as e:
        logger.warning("Cleanup failed for %s: %s", filepath, e)


@app.errorhandler(413)
def too_large(e):
    max_mb = MAX_CONTENT_LENGTH // (1024 * 1024)
    return jsonify({"success": False, "error": f"File too large. Maximum size is {max_mb} MB."}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"success": False, "error": "Internal server error. Please try again."}), 500


if __name__ == "__main__":
    logger.info("Starting Enhanced Deepfake Detector v2 …")
    app.run(debug=True, threaded=True)
