import cv2

from landmarks.face_landmarks import FaceLandmarks
from segmentation.region_masks import RegionMasks
from segmentation.landmark_indices import (
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_IRIS,
    RIGHT_IRIS,
    MOUTH
)


image = cv2.imread("test.jpeg")

if image is None:
    print("Could not load test.jpeg")
    exit()


# --------------------------------
# Detect landmarks
# --------------------------------

detector = FaceLandmarks()

faces = detector.detect(image)

if len(faces) == 0:

    print("No face detected")
    exit()


landmarks = faces[0]


# --------------------------------
# Create mask generator
# --------------------------------

mask_generator = RegionMasks(
    image.shape
)


# --------------------------------
# Create masks
# --------------------------------

left_eye_mask = mask_generator.create_eye_mask(
    landmarks,
    LEFT_EYE
)

right_eye_mask = mask_generator.create_eye_mask(
    landmarks,
    RIGHT_EYE
)

left_iris_mask = mask_generator.create_iris_mask(
    landmarks,
    LEFT_IRIS
)

right_iris_mask = mask_generator.create_iris_mask(
    landmarks,
    RIGHT_IRIS
)

mouth_mask = mask_generator.create_mouth_mask(
    landmarks,
    MOUTH
)


# --------------------------------
# Save masks
# --------------------------------

cv2.imwrite(
    "left_eye_mask.png",
    left_eye_mask
)

cv2.imwrite(
    "right_eye_mask.png",
    right_eye_mask
)

cv2.imwrite(
    "left_iris_mask.png",
    left_iris_mask
)

cv2.imwrite(
    "right_iris_mask.png",
    right_iris_mask
)

cv2.imwrite(
    "mouth_mask.png",
    mouth_mask
)


print("Masks generated successfully.")