"""
@file app.py
@brief Image Forgery Detection System - Main Flask Application
@author ImageForensics Team
@version 1.0.0
@date 2026

@mainpage Image Forgery Detection System

@section intro_sec Introduction
This system detects image manipulation using multiple computer vision algorithms:
SIFT, SURF, AKAZE, ORB, and VGG-based feature matching.

@section install_sec Installation
Run: pip install -r requirements.txt

@section usage_sec Usage
python app.py
"""

import os
import sys
import base64
import logging
import numpy as np
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from flask import send_from_directory

# Import forensics engine
from forensics_engine import ForensicsEngine

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

engine = ForensicsEngine()

# ---------------------------------------------------------------------------
# FSM States (used for effort/state tracking)
# ---------------------------------------------------------------------------
# States: IDLE -> UPLOAD -> PREPROCESS -> ANALYZE -> REPORT -> IDLE
FSM_STATE = "IDLE"
FSM_TRANSITIONS = {
    "IDLE": ["UPLOAD"],
    "UPLOAD": ["PREPROCESS"],
    "PREPROCESS": ["ANALYZE"],
    "ANALYZE": ["REPORT"],
    "REPORT": ["IDLE"]
}


def fsm_transition(current: str, next_state: str) -> str:
    """
    @brief Perform FSM state transition with validation.
    @param current Current FSM state string.
    @param next_state Desired next FSM state string.
    @return New state string if transition is valid.
    @throws ValueError if the transition is not allowed.
    """
    if next_state not in FSM_TRANSITIONS.get(current, []):
        raise ValueError(f"Invalid FSM transition: {current} -> {next_state}")
    logger.info(f"FSM: {current} -> {next_state}")
    return next_state


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(".", "main.html")


@app.route("/api/health", methods=["GET"])
def health():
    """
    @brief Health check endpoint.
    @return JSON with status 'ok'.
    """
    return jsonify({"status": "ok", "version": "1.0.0"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    @brief Main analysis endpoint. Accepts an image and runs all forensics algorithms.

    @details
    FSM Flow: IDLE -> UPLOAD -> PREPROCESS -> ANALYZE -> REPORT -> IDLE

    Expects multipart/form-data with:
    - file: image file (jpg, png, bmp, tiff, webp)
    - algorithms: comma-separated list (sift,surf,akaze,orb)

    @return JSON with full forensics report including:
            - tampered (bool)
            - confidence (float 0-100)
            - ai_generated (bool)
            - ai_confidence (float 0-100)
            - algorithm_results (dict)
            - heatmap_b64 (str, base64 PNG)
            - annotated_b64 (str, base64 PNG)
            - ela_b64 (str, base64 PNG)
    """
    global FSM_STATE
    FSM_STATE = "IDLE"

    # --- UPLOAD state ---
    try:
        FSM_STATE = fsm_transition(FSM_STATE, "UPLOAD")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    algorithms_param = request.form.get("algorithms", "sift,surf,akaze,orb")
    selected_algos = [a.strip().lower() for a in algorithms_param.split(",")]

    try:
        img_bytes = file.read()
        pil_raw = Image.open(BytesIO(img_bytes))
        fmt = getattr(pil_raw, "format", "") or ""
        is_gif = fmt.upper() == "GIF"
        is_animated_gif = is_gif and getattr(pil_raw, "n_frames", 1) > 1
        n_frames = getattr(pil_raw, "n_frames", 1) if is_gif else 1
        if is_gif:
            pil_raw.seek(0)
        pil_img = pil_raw.convert("RGB")
    except Exception as e:
        logger.error(f"Image load failed: {e}")
        return jsonify({"error": "Cannot read image file"}), 400

    # --- PREPROCESS state ---
    FSM_STATE = fsm_transition(FSM_STATE, "PREPROCESS")
    img_np = np.array(pil_img)

    # --- ANALYZE state ---
    FSM_STATE = fsm_transition(FSM_STATE, "ANALYZE")
    is_gif = locals().get("is_gif", False)
    is_animated_gif = locals().get("is_animated_gif", False)
    n_frames = locals().get("n_frames", 1)
    try:
        result = engine.run_full_analysis(img_np, img_bytes, selected_algos)
        if is_gif:
            result["file_format"] = "GIF"
            result["gif_animated"] = is_animated_gif
            result["gif_frames"] = n_frames
    except Exception as e:
        logger.exception("Analysis failed")
        return jsonify({"error": f"Analysis error: {str(e)}"}), 500

    # --- REPORT state ---
    FSM_STATE = fsm_transition(FSM_STATE, "REPORT")
    FSM_STATE = fsm_transition(FSM_STATE, "IDLE")

    return jsonify(result)


@app.route("/api/algorithms", methods=["GET"])
def list_algorithms():
    """
    @brief Returns available algorithms and their descriptions.
    @return JSON list of algorithm objects.
    """
    algos = [
        {"id": "sift",  "name": "SIFT",  "full": "Scale-Invariant Feature Transform",
         "description": "Kopyala-yapıştır manipülasyonlarını tespit eder. Döndürme ve ölçek değişimlerine karşı dayanıklıdır."},
        {"id": "surf",  "name": "SURF",  "full": "Speeded-Up Robust Features",
         "description": "SIFT'in hızlandırılmış versiyonu. Doku manipülasyonlarında etkilidir."},
        {"id": "akaze", "name": "AKAZE", "full": "Accelerated KAZE",
         "description": "Doğrusal olmayan difüzyon filtresi tabanlı. Yüksek doğruluk sağlar."},
        {"id": "orb",   "name": "ORB",   "full": "Oriented FAST and Rotated BRIEF",
         "description": "Hızlı ve açık kaynaklı. Gerçek zamanlı analiz için uygundur."},
    ]
    return jsonify(algos)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting Image Forensics Server on http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)