import cv2
import numpy as np

from landmarks.face_landmarks import FaceLandmarks
from enhancement.eye_bag_mask import EyeBagMask
from enhancement.eye_bag import EyeBagEnhancer
from segmentation.landmark_indices import (
    LEFT_EYE,
    RIGHT_EYE
)


IMAGE_PATH = "test.jpeg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise RuntimeError("Could not load image")


# -------------------------
# 1. Get face landmarks
# -------------------------

landmark_detector = FaceLandmarks()

landmarks = landmark_detector.get_points(image)

if landmarks is None:
    raise RuntimeError("No face landmarks detected")


# -------------------------
# 2. Create eye-bag masks
# -------------------------

mask_generator = EyeBagMask(image.shape)

left_mask = mask_generator.create(
    landmarks,
    LEFT_EYE
)

right_mask = mask_generator.create(
    landmarks,
    RIGHT_EYE
)

eye_bag_mask = cv2.bitwise_or(
    left_mask,
    right_mask
)


# -------------------------
# 3. Save mask
# -------------------------

cv2.imwrite(
    "eye_bag_mask.png",
    eye_bag_mask
)


# -------------------------
# 4. Debug visualization
# -------------------------

debug = image.copy()

overlay = np.zeros_like(image)

overlay[:, :, 2] = eye_bag_mask

debug = cv2.addWeighted(
    debug,
    0.7,
    overlay,
    0.3,
    0
)

cv2.imwrite(
    "eye_bag_mask_debug.jpg",
    debug
)


# -------------------------
# 5. Enhancement
# -------------------------

enhancer = EyeBagEnhancer()

for strength in [0.25, 0.5, 0.75, 1.0]:

    result = enhancer.reduce(
        image,
        eye_bag_mask,
        strength=strength
    )

    output_path = (
        f"eye_bags_{int(strength * 100)}.jpg"
    )

    cv2.imwrite(
        output_path,
        result
    )

    print("Generated:", output_path)


print("Eye-bag reduction completed.")