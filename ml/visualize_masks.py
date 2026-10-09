import cv2
import numpy as np

from landmarks.face_landmarks import FaceLandmarks
from segmentation.region_masks import RegionMasks
from segmentation.landmark_indices import (
    LEFT_EYE,
    RIGHT_EYE,
    MOUTH
)


image = cv2.imread("test.jpeg")

detector = FaceLandmarks()

faces = detector.detect(image)

if not faces:
    print("No face detected")
    exit()

landmarks = faces[0]

mask_generator = RegionMasks(image.shape)

left_eye = mask_generator.create_eye_mask(
    landmarks,
    LEFT_EYE
)

right_eye = mask_generator.create_eye_mask(
    landmarks,
    RIGHT_EYE
)

mouth = mask_generator.create_mouth_mask(
    landmarks,
    MOUTH
)


# Create overlay

overlay = image.copy()

overlay[left_eye > 0] = (
    0,
    255,
    0
)

overlay[right_eye > 0] = (
    0,
    255,
    0
)

overlay[mouth > 0] = (
    255,
    0,
    0
)


# Blend

result = cv2.addWeighted(
    image,
    0.7,
    overlay,
    0.3,
    0
)


cv2.imwrite(
    "regions.jpg",
    result
)

print("Saved regions.jpg")