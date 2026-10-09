import cv2

from landmarks.face_landmarks import FaceLandmarks
from segmentation.region_masks import RegionMasks
from segmentation.landmark_indices import (
    LEFT_IRIS,
    RIGHT_IRIS
)

from enhancement.iris import IrisEnhancer


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
# Create mask generator
# --------------------------------

mask_generator = RegionMasks(
    image.shape
)


# --------------------------------
# Create iris masks
# --------------------------------

left_iris_mask = mask_generator.create_iris_mask(
    landmarks,
    LEFT_IRIS
)

right_iris_mask = mask_generator.create_iris_mask(
    landmarks,
    RIGHT_IRIS
)


# --------------------------------
# Combine both iris masks
# --------------------------------

iris_mask = cv2.bitwise_or(
    left_iris_mask,
    right_iris_mask
)


# --------------------------------
# Enhance iris
# --------------------------------

enhancer = IrisEnhancer()

result = enhancer.enhance(
    image,
    iris_mask,
    strength=0.5
)


# --------------------------------
# Save results
# --------------------------------

cv2.imwrite(
    "iris_enhanced.jpg",
    result
)

cv2.imwrite(
    "iris_mask.jpg",
    iris_mask
)


print("Iris enhancement completed.")