import threading
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from pipeline import FaceAwarePipeline

MODEL_PATH = "../models/face_parser/resnet18.pt"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_SIDE = 2048

state = {}
lock = threading.Lock()  # pipeline (MediaPipe / torch) is not safe to share across threads


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["pipeline"] = FaceAwarePipeline(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(title="FaceAware AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/enhance")
def enhance(
    image: UploadFile = File(...),
    skin_strength: float = Form(0.0, ge=0.0, le=1.0),
    eye_strength: float = Form(0.0, ge=0.0, le=1.0),
    iris_strength: float = Form(0.0, ge=0.0, le=1.0),
    eye_bag_strength: float = Form(0.0, ge=0.0, le=1.0),
    teeth_strength: float = Form(0.0, ge=0.0, le=1.0),
    glasses_strength: float = Form(0.0, ge=0.0, le=1.0),
    wrinkle_strength: float = Form(0.0, ge=0.0, le=1.0),
):
    data = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large (max 15 MB)")

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image file")

    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

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

    ok, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise HTTPException(500, "Failed to encode result")

    return Response(content=buf.tobytes(), media_type="image/jpeg")