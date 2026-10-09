import cv2
import numpy as np

from landmarks.face_landmarks import FaceLandmarks
from segmentation.region_masks import RegionMasks
from segmentation.landmark_indices import (
    MOUTH,
    INNER_MOUTH
)
from enhancement.teeth import TeethEnhancer


# --------------------------------
# Load image
# --------------------------------

image = cv2.imread("test2.jpg")

if image is None:
    print("Could not load test2.jpg")
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
# Create mouth masks
# --------------------------------

mask_generator = RegionMasks(
    image.shape
)

mouth_mask = mask_generator.create_mouth_mask(
    landmarks,
    MOUTH
)

inner_mouth_mask = mask_generator.create_inner_mouth_mask(
    landmarks,
    INNER_MOUTH
)


# --------------------------------
# Convert image to LAB
# --------------------------------

lab = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2LAB
)

L, A, B = cv2.split(lab)


# --------------------------------
# Convert image to HSV
# --------------------------------

hsv = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2HSV
)

H, S, V = cv2.split(hsv)


# --------------------------------
# Restrict analysis to inner mouth
# --------------------------------

mouth_region = inner_mouth_mask > 0

mouth_L = L[mouth_region]

if len(mouth_L) == 0:
    print("Inner mouth region is empty")
    exit()


# --------------------------------
# Adaptive brightness threshold
# --------------------------------

L_threshold = np.percentile(
    mouth_L,
    55
)


# --------------------------------
# Teeth characteristics
# --------------------------------

bright = L > L_threshold

# Teeth are usually less saturated than lips
low_saturation = S < 150

# Remove very dark mouth interior
not_dark = L > 80


candidate = (
    bright &
    low_saturation &
    not_dark &
    mouth_region
).astype(np.uint8) * 255


# --------------------------------
# Morphological cleanup
# --------------------------------

kernel = np.ones(
    (3, 3),
    dtype=np.uint8
)

teeth_mask = cv2.morphologyEx(
    candidate,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1
)

teeth_mask = cv2.morphologyEx(
    teeth_mask,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)


# --------------------------------
# Connected components
# --------------------------------

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    teeth_mask,
    connectivity=8
)

clean_mask = np.zeros_like(teeth_mask)

for i in range(1, num_labels):

    area = stats[i, cv2.CC_STAT_AREA]

    if area > 20:
        clean_mask[labels == i] = 255


teeth_mask = clean_mask


# --------------------------------
# Save teeth mask
# --------------------------------

cv2.imwrite(
    "teeth_mask_refined.png",
    teeth_mask
)


# --------------------------------
# Whitening
# --------------------------------

enhancer = TeethEnhancer()

result = enhancer.whiten(
    image,
    teeth_mask,
    strength=0.5
)


# --------------------------------
# Save result
# --------------------------------

cv2.imwrite(
    "inner_mouth_mask.png",
    inner_mouth_mask
)

cv2.imwrite(
    "teeth_whitened_refined.jpg",
    result
)

cv2.imwrite(
    "teeth_candidate.png",
    candidate
)


print("Refined teeth mask generated.")
print("Teeth whitening completed.")