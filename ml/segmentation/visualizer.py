import cv2
import numpy as np


def create_overlay(image, segmentation):
    """
    Create a colored visualization of the face parsing result.
    """

    overlay = image.copy()

    # Face parsing class IDs
    colors = {
        1: (0, 255, 0),       # Skin
        2: (255, 0, 0),       # Left eyebrow
        3: (255, 0, 0),       # Right eyebrow
        4: (0, 255, 255),     # Left eye
        5: (0, 255, 255),     # Right eye
        6: (255, 0, 255),     # Glasses
        7: (0, 128, 255),      # Left ear
        8: (0, 128, 255),      # Right ear
        9: (128, 0, 255),      # Earring
        10: (255, 128, 0),     # Nose
        11: (0, 0, 255),       # Mouth
        12: (0, 0, 255),       # Upper lip
        13: (0, 0, 255),       # Lower lip
        14: (128, 128, 0),     # Neck
        15: (128, 0, 128),     # Necklace
        16: (128, 128, 128),   # Cloth
        17: (255, 128, 128),   # Hair
        18: (128, 255, 128),   # Hat
    }

    for class_id, color in colors.items():

        mask = segmentation == class_id

        overlay[mask] = color

    # Blend original image and segmentation
    result = cv2.addWeighted(
        image,
        0.55,
        overlay,
        0.45,
        0
    )

    return result