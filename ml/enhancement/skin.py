import cv2
import numpy as np


def smooth_skin(
    image,
    skin_mask,
    strength=0.5
):

    # Make sure strength is valid
    strength = np.clip(
        strength,
        0.0,
        1.0
    )


    # Create smoothed version
    smoothed = cv2.bilateralFilter(
        image,
        d=9,
        sigmaColor=75,
        sigmaSpace=75
    )


    # Convert mask to float
    mask = (
        skin_mask.astype(
            np.float32
        ) / 255.0
    )


    # Apply strength
    alpha = (
        mask * strength
    )


    # Expand mask to 3 channels
    alpha = np.expand_dims(
        alpha,
        axis=2
    )


    # Blend
    result = (
        image.astype(np.float32)
        * (1 - alpha)
        +
        smoothed.astype(np.float32)
        * alpha
    )


    result = np.clip(
        result,
        0,
        255
    ).astype(
        np.uint8
    )


    return result