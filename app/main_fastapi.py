"""
============================================================
Facial Skin Condition Detection — FastAPI Backend
(Open-Set Rejection & Multi-Label Classification)
============================================================

Endpoints:
  GET  /        -> Welcome & API overview
  GET  /health  -> Health status, active models, supported classes
  POST /predict -> Single facial image evaluation with rejection handling

Run:
  uvicorn app.main_fastapi:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import uuid
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# -- Path Setup ---------------------------------------------------------------

APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR      = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

try:
    from predict import predict_single_image, SUPPORTED_CLASSES
    _MODEL_AVAILABLE = True
    _IMPORT_ERROR    = None
except Exception as e:
    _MODEL_AVAILABLE = False
    _IMPORT_ERROR    = str(e)
    SUPPORTED_CLASSES = ["Acne", "Melasma", "Rosacea", "Seborrheic Dermatitis"]

# -- FastAPI App Setup --------------------------------------------------------

app = FastAPI(
    title="DermScan AI — Facial Skin Condition Detection API",
    description="FastAPI service for facial skin condition detection with open-set rejection.",
    version="2.0.0",
)

# Enable CORS for React and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


# -- Schemas ------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model_available: bool
    supported_classes: List[str]
    version: str


# -- Routes -------------------------------------------------------------------

@app.get("/", tags=["General"])
async def root():
    return {
        "service": "DermScan AI Facial Skin Condition API",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    return HealthResponse(
        status="ok" if _MODEL_AVAILABLE else "degraded",
        model_available=_MODEL_AVAILABLE,
        supported_classes=SUPPORTED_CLASSES,
        version="2.0.0",
    )


@app.post("/predict", tags=["Inference"])
async def predict_endpoint(file: UploadFile = File(...)):
    # 1. Filename validation
    filename = file.filename or ""
    if not filename:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "is_rejected": True,
                "error_code": "NO_FILE",
                "error": "No file uploaded. Please upload a clear image containing one face.",
                "message": "Please upload a clear image containing one face.",
            },
        )

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "is_rejected": True,
                "error_code": "UNSUPPORTED_FORMAT",
                "error": f"Unsupported format (.{ext}). Allowed formats: JPG, JPEG, PNG, WEBP, BMP.",
                "message": "Unable to classify this image.",
            },
        )

    if not _MODEL_AVAILABLE:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "success": False,
                "is_rejected": True,
                "error_code": "MODEL_UNAVAILABLE",
                "error": f"Model pipeline is not ready: {_IMPORT_ERROR}",
                "message": "Unable to classify this image.",
            },
        )

    # 2. Save temporary upload safely
    suffix = f".{ext}"
    tmp_path = Path(tempfile.gettempdir()) / f"fastapi_skin_{uuid.uuid4().hex}{suffix}"

    try:
        content = await file.read()
        if len(content) == 0:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "is_rejected": True,
                    "error_code": "EMPTY_FILE",
                    "error": "Uploaded file is 0 bytes.",
                    "message": "Unable to classify this image.",
                },
            )

        if len(content) > MAX_FILE_SIZE_BYTES:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "is_rejected": True,
                    "error_code": "FILE_TOO_LARGE",
                    "error": "File exceeds maximum upload size (10 MB).",
                    "message": "Unable to classify this image.",
                },
            )

        with open(tmp_path, "wb") as f:
            f.write(content)

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "is_rejected": True,
                "error_code": "SAVE_FAILED",
                "error": f"Failed to process uploaded file: {str(e)}",
                "message": "Unable to classify this image.",
            },
        )

    # 3. Execute Prediction & Rejection
    try:
        result = predict_single_image(tmp_path, original_filename=filename)
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

    # 4. Handle Pre-Validation Rejections & Server/Pipeline Errors
    if not result.get("success", False):
        err_code = result.get("error_code", "VALIDATION_FAILED")
        user_message = result.get("message", "Invalid image for facial skin analysis.")
        title = "Invalid Image"
        line1 = result.get("line1") or user_message
        line2 = result.get("line2") or "Please upload a clear image of a single face showing a facial skin condition."

        if err_code == "MULTIPLE_FACES_DETECTED":
            title = "Invalid Image"
            line1 = "Invalid image: Multiple faces were detected in this image."
            line2 = "Please upload an image with only one face showing a skin condition."
        elif err_code in ("NO_FACE_DETECTED", "NO_FACE_INFECTION_DETECTED"):
            title = "Invalid Image"
            line1 = "No facial skin infection was detected in this image."
            line2 = "Please upload a clear facial skin image."
        elif err_code == "INVALID_IMAGE":
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

        # If it's a server or model missing error, return 500 or 503
        http_status = status.HTTP_500_INTERNAL_SERVER_ERROR if err_code in ("MODEL_MISSING", "MODEL_LOAD_ERROR", "INVALID_MODEL", "PREDICTION_ERROR") else status.HTTP_400_BAD_REQUEST

        return JSONResponse(
            status_code=http_status,
            content={
                "success": False,
                "is_rejected": True,
                "error_code": err_code,
                "title": title,
                "line1": line1,
                "line2": line2,
                "error": user_message,
                "message": user_message,
            },
        )

    # 5. Handle In-Model Rejection (Unsupported skin condition outside supported categories)
    if result.get("is_rejected", False):
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "is_rejected": True,
                "rejection_code": result.get("rejection_code", "OUTSIDE_SUPPORTED_CATEGORIES"),
                "badge": result.get("badge", "Unable to Classify"),
                "message": result.get("message", "This skin condition may be outside the supported categories."),
                "detailed_reason": result.get("detailed_reason", "We couldn't detect a known skin disease from our model. Please try another image or consult a dermatologist."),
                "predicted_class": result.get("predicted_class", "Unsupported / Unknown Condition"),
                "is_clear_skin": False,
                "is_other_disease": True,
                "is_multi_condition": False,
                "detected_conditions": result.get("detected_conditions", []),
                "decision_metrics": result.get("decision_metrics", {}),
                "affected_area": result.get("affected_area", {}),
                "condition_info": result.get("condition_info", {}),
                "prescription": result.get("prescription", {}),
                "disclaimer": result.get("disclaimer", ""),
            },
        )

    # 6. Accepted Classification -> HTTP 200 with full disease details
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "success": True,
            "is_rejected": False,
            "is_clear_skin": result.get("is_clear_skin", False),
            "is_other_disease": False,
            "is_multi_condition": result.get("is_multi_condition", False),
            "detected_conditions": result.get("detected_conditions", []),
            "predicted_class": result.get("predicted_class"),
            "confidence": result.get("confidence"),
            "probabilities": result.get("probabilities", {}),
            "decision_metrics": result.get("decision_metrics", {}),
            "affected_area": result.get("affected_area", {}),
            "condition_info": result.get("condition_info", {}),
            "prescription": result.get("prescription", {}),
            "disclaimer": result.get("disclaimer", ""),
        },
    )


# -- Entry Point --------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print("\n" + "=" * 60)
    print("  DermScan AI — FastAPI Skin Condition Detection Service")
    print(f"  http://localhost:{port} (API Docs: http://localhost:{port}/docs)")
    print("=" * 60 + "\n")
    uvicorn.run("app.main_fastapi:app", host="0.0.0.0", port=port, reload=True)
