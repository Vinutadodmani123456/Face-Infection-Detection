"""
============================================================
Facial Skin Condition Detection — Flask Web Application
(Multi-Label Enhanced)
============================================================

Routes
------
GET  /           -> Render main UI (index.html)
POST /predict    -> Accept uploaded image, return multi-label prediction JSON
GET  /health     -> Health check endpoint

Run:
    cd app
    python app.py
Then open: http://localhost:5050

Author: MCA Project
"""

import os
import sys
import uuid
import tempfile
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# -- Path Setup ---------------------------------------------------------------

APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR      = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

try:
    from predict import predict_single_image, validate_face
    _MODEL_AVAILABLE = True
except ImportError as e:
    _MODEL_AVAILABLE = False
    _IMPORT_ERROR    = str(e)

# -- Flask App ----------------------------------------------------------------

app = Flask(__name__, template_folder="templates", static_folder="static")

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB max upload
app.config["SECRET_KEY"]         = os.urandom(24)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# -- Routes -------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_available": _MODEL_AVAILABLE})


@app.route("/predict", methods=["POST"])
def predict():
    # -- File validation ------------------------------------------------------
    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image uploaded. Please select a facial image.",
            "error_code": "NO_FILE",
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "No file selected. Please choose an image to upload.",
            "error_code": "EMPTY_FILE",
        }), 400

    if not _allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported file format. Please upload JPG, JPEG, PNG, WebP, or BMP.",
            "error_code": "UNSUPPORTED_FORMAT",
        }), 400

    if not _MODEL_AVAILABLE:
        return jsonify({
            "success": False,
            "error": (
                "Model not available. Please run the training pipeline first:\n"
                "1. python src/feature_extraction.py\n"
                "2. python src/train_models.py"
            ),
            "error_code": "MODEL_MISSING",
        }), 503

    # -- Save to temp file ----------------------------------------------------
    suffix   = "." + secure_filename(file.filename).rsplit(".", 1)[-1].lower()
    tmp_path = Path(tempfile.gettempdir()) / f"skin_{uuid.uuid4().hex}{suffix}"

    try:
        file.save(str(tmp_path))
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to save file: {e}",
            "error_code": "SAVE_FAILED",
        }), 500

    # -- Predict (multi-label with open-set rejection) ------------------------
    try:
        result = predict_single_image(tmp_path, original_filename=file.filename)
    except Exception as e:
        result = {
            "success": False,
            "is_rejected": True,
            "error_code": "PREDICTION_EXCEPTION",
            "error": str(e),
            "message": "Unable to classify this image.",
        }
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    # -- Handle Pre-Validation Rejections (HTTP 400) -------------------------
    if not result.get("success", False):
        err_code = result.get("error_code", "INVALID_IMAGE")
        user_message = result.get("message", "Invalid image for facial skin analysis.")
        title = "Invalid Image"
        line1 = result.get("line1") or user_message
        line2 = result.get("line2") or "Please upload a clear image of a single face showing a facial skin condition."

        if err_code == "NO_FACE_INFECTION_DETECTED":
            title = "Invalid Image"
            line1 = "No facial skin infection was detected in this image."
            line2 = "Please upload a clear image of a face showing a skin condition."
        elif err_code == "MULTIPLE_FACES_DETECTED":
            title = "Invalid Image"
            line1 = "Invalid image: Multiple faces were detected in this image."
            line2 = "Please upload an image with only one face showing a skin condition."
        elif err_code in ("NO_FACE_DETECTED", "INVALID_IMAGE"):
            title = "Invalid Image"
            line1 = "This image is invalid for facial skin infection analysis."
            line2 = "Please upload a clear image of a single face showing a facial skin condition."
        elif err_code == "IMAGE_TOO_BLURRY":
            title = "Invalid Image"
            line1 = "Image is too blurry for reliable dermatological assessment."
            line2 = "Please upload a sharper, in-focus image."
        elif err_code in ("IMAGE_TOO_DARK", "IMAGE_TOO_BRIGHT"):
            title = "Invalid Image"
            line1 = user_message
            line2 = "Please upload an image with balanced lighting."

        http_code = 500 if err_code in ("MODEL_MISSING", "MODEL_LOAD_ERROR", "INVALID_MODEL", "PREDICTION_ERROR") else 400

        return jsonify({
            "success": False,
            "is_rejected": True,
            "error_code": err_code,
            "title": title,
            "line1": line1,
            "line2": line2,
            "error": user_message,
            "message": user_message,
        }), http_code

    # -- Handle In-Model Rejection (Unsupported Condition) --------------------
    if result.get("is_rejected", False):
        return jsonify({
            "success":              True,
            "is_rejected":          True,
            "rejection_code":       result.get("rejection_code", "OUTSIDE_SUPPORTED_CATEGORIES"),
            "badge":                result.get("badge", "Unknown Facial Condition"),
            "message":              result.get("message", "This skin condition may be outside the supported categories."),
            "detailed_reason":      result.get("detailed_reason", "We couldn't detect a known skin disease from our model. Please try another image or consult a dermatologist."),
            "predicted_class":      result.get("predicted_class", "Unknown Facial Condition"),
            "title":                result.get("title", "Unknown Facial Condition"),
            "is_clear_skin":        False,
            "is_other_disease":     True,
            "is_multi_condition":   False,
            "detected_conditions":  result.get("detected_conditions", []),
            "probabilities":        result.get("probabilities", {"Acne": 0.0, "Melasma": 0.0, "Rosacea": 0.0, "Seborrheic Dermatitis": 0.0}),
            "decision_metrics":     result.get("decision_metrics", {}),
            "affected_area":        result.get("affected_area", {}),
            "prescription":         result.get("prescription", {}),
            "condition_info":       result.get("condition_info", {}),
            "disclaimer":           result.get("disclaimer", ""),
        }), 200

    # -- Serialise and return accepted classification -------------------------
    response = {
        "success":              True,
        "is_rejected":          False,
        "is_clear_skin":        result.get("is_clear_skin", False),
        "is_other_disease":     False,
        "is_multi_condition":   result.get("is_multi_condition", False),
        "detected_conditions":  result.get("detected_conditions", []),
        "predicted_class":      result["predicted_class"],
        "confidence":           result.get("confidence", 0.0),
        "probabilities":        result.get("probabilities", {}),
        "decision_metrics":     result.get("decision_metrics", {}),
        "affected_area":        result.get("affected_area", {
            "percentage": 25.0,
            "severity": "Moderate",
            "severity_class": "severity-moderate"
        }),
        "prescription":         result.get("prescription", {}),
        "condition_info":       result.get("condition_info", {}),
        "disclaimer":           result.get("disclaimer", ""),
    }
    return jsonify(response), 200


# -- Entry Point --------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print("\n" + "=" * 60)
    print("  DermScan AI — Multi-Label Facial Skin Condition Detection")
    print(f"  http://localhost:{port}")
    print("=" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=port)
