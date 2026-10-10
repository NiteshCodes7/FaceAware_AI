import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from pipeline import FaceAwarePipeline


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_PATH = (
    BASE_DIR.parent
    / "models"
    / "face_parser"
    / "training"
    / "bisenet_celebamask_best.pt"
)

# Overridden in Docker via the MODEL_PATH env var
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if o.strip()
]

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_SIDE = 2048


# ---------------------------------------------------------
# Global state
# ---------------------------------------------------------

state = {}

# MediaPipe / PyTorch pipeline is shared,
# so protect it from concurrent requests.
lock = threading.Lock()


# ---------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing FaceAware AI pipeline...")
    print(f"Model: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    state["pipeline"] = FaceAwarePipeline(str(MODEL_PATH))

    print("FaceAware AI pipeline initialized.")

    yield

    state.clear()
    print("FaceAware AI pipeline stopped.")


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="FaceAware AI",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH.name,
    }


# ---------------------------------------------------------
# Image enhancement
# ---------------------------------------------------------

@app.post("/enhance")
def enhance(
    image: UploadFile = File(...),

    skin_strength: float = Form(
        0.0, ge=0.0, le=1.0
    ),

    eye_strength: float = Form(
        0.0, ge=0.0, le=1.0
    ),

    iris_strength: float = Form(
        0.0, ge=0.0, le=1.0
    ),

    eye_bag_strength: float = Form(
        0.0, ge=0.0, le=1.0
    ),

    teeth_strength: float = Form(
        0.0, ge=0.0, le=1.0
    ),

    glasses_strength: float = Form(
        0.0, ge=0.0, le=1.0
    ),

    wrinkle_strength: float = Form(
        0.0, ge=0.0, le=1.0
    ),
):

    # -----------------------------------------------------
    # Read uploaded image
    # -----------------------------------------------------

    data = image.file.read(MAX_UPLOAD_BYTES + 1)

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image too large (max 15 MB)",
        )

    # -----------------------------------------------------
    # Decode image
    # -----------------------------------------------------

    img = cv2.imdecode(
        np.frombuffer(data, np.uint8),
        cv2.IMREAD_COLOR,
    )

    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file",
        )

    # -----------------------------------------------------
    # Limit maximum image dimension
    # -----------------------------------------------------

    h, w = img.shape[:2]

    scale = MAX_SIDE / max(h, w)

    if scale < 1:
        img = cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    # -----------------------------------------------------
    # Run FaceAware pipeline
    # -----------------------------------------------------

    with lock:

        result = state["pipeline"].process(
            img,

            skin_strength=skin_strength,
            eye_strength=eye_strength,
            iris_strength=iris_strength,
            eye_bag_strength=eye_bag_strength,
            teeth_strength=teeth_strength,
            glasses_strength=glasses_strength,
            wrinkle_strength=wrinkle_strength,
        )

    # -----------------------------------------------------
    # Encode output as JPEG
    # -----------------------------------------------------

    ok, buf = cv2.imencode(
        ".jpg",
        result,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    if not ok:
        raise HTTPException(
            status_code=500,
            detail="Failed to encode result",
        )

    return Response(
        content=buf.tobytes(),
        media_type="image/jpeg",
    )