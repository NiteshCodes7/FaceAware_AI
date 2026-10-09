import cv2

from landmarks.face_landmarks import FaceLandmarks

from segmentation.region_masks import RegionMasks

from segmentation.landmark_indices import (
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_IRIS,
    RIGHT_IRIS
)

from enhancement.eyes import EyeEnhancer


# --------------------------------
# Load image
# --------------------------------

image = cv2.imread("test.jpeg")

if image is None:
    print("Could not load test.jpeg")
    exit()


# --------------------------------
# Detect landmarks
# --------------------------------

detector = FaceLandmarks()

landmarks = detector.get_points(image)

if landmarks is None:
    print("No face detected")
    exit()


# --------------------------------
# Generate masks
# --------------------------------

mask_generator = RegionMasks(
    image.shape
)

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


# --------------------------------
# Create enhancer
# --------------------------------

enhancer = EyeEnhancer()


# --------------------------------
# Create sclera masks
# --------------------------------

left_sclera = enhancer.create_sclera_mask(
    left_eye_mask,
    left_iris_mask
)

right_sclera = enhancer.create_sclera_mask(
    right_eye_mask,
    right_iris_mask
)


# --------------------------------
# Combine both eyes
# --------------------------------

sclera_mask = cv2.bitwise_or(
    left_sclera,
    right_sclera
)


# --------------------------------
# Enhance eyes
# --------------------------------

result = enhancer.whiten(
    image,
    sclera_mask,
    strength=0.5
)


# --------------------------------
# Save result
# --------------------------------

cv2.imwrite(
    "eye_whitened.jpg",
    result
)

cv2.imwrite(
    "sclera_mask.jpg",
    sclera_mask
)


print("Eye whitening completed.")