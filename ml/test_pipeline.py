import os
import cv2

from pipeline import FaceAwarePipeline

# ============================================================
# CONFIGURATION (single source of truth)
# ============================================================

IMAGE_PATH = "test.jpeg"         
MODEL_PATH = "../models/face_parser/resnet18.pt"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# All strengths default to 0, tests override only what they need
ZERO = dict(
    skin_strength=0.0,
    eye_bag_strength=0.0,
    eye_strength=0.0,
    iris_strength=0.0,
    teeth_strength=0.0,
    glasses_strength=0.0,
    wrinkle_strength=0.0,
)

TESTS = {
    "original":   {},
    "skin_50":    dict(skin_strength=0.5),
    "eye_50":     dict(eye_strength=0.5),
    "iris_50":    dict(iris_strength=0.5),
    "eye_bag_50": dict(eye_bag_strength=0.5),
    "teeth_50":   dict(teeth_strength=0.5),
    "wrinkles_50": dict(wrinkle_strength=0.5),
    "wrinkles_100": dict(wrinkle_strength=1.0),
    "glasses_25":  dict(glasses_strength=0.25),
    "glasses_50":  dict(glasses_strength=0.5),
    "glasses_75":  dict(glasses_strength=0.75),
    "glasses_100": dict(glasses_strength=1.0),
    "combined": dict(
        skin_strength=0.4,
        eye_bag_strength=0.4,
        eye_strength=0.4,
        iris_strength=0.4,
        teeth_strength=0.5,
        wrinkle_strength=0.5,
    ),
    "full": dict(
        skin_strength=0.4,
        eye_bag_strength=0.4,
        eye_strength=0.4,
        iris_strength=0.4,
        teeth_strength=0.5,
        glasses_strength=0.8,
        wrinkle_strength=0.5,
    ),
}

# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise RuntimeError(f"Could not load image: {IMAGE_PATH}")

print(f"Image loaded: {IMAGE_PATH} {image.shape}")

# ============================================================
# INITIALIZE PIPELINE (once)
# ============================================================

pipeline = FaceAwarePipeline(MODEL_PATH)

# ============================================================
# RUN TESTS
# ============================================================

for name, overrides in TESTS.items():

    print(f"\nTesting {name}...")

    params = {**ZERO, **overrides}
    result = pipeline.process(image, **params)

    out = os.path.join(OUTPUT_DIR, f"pipeline_{name}.jpg")
    cv2.imwrite(out, result)

    print("Generated:", out)

print("\n======================================")
print("FaceAware Pipeline testing completed.")
print("======================================")