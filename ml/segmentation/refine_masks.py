import cv2
import numpy as np


def refine_skin_mask(
    skin,
    eyes,
    eyebrows,
    mouth,
    lips,
    hair
):

    refined = skin.copy()


    excluded = cv2.bitwise_or(
        eyes,
        eyebrows
    )

    excluded = cv2.bitwise_or(
        excluded,
        mouth
    )

    excluded = cv2.bitwise_or(
        excluded,
        lips
    )

    excluded = cv2.bitwise_or(
        excluded,
        hair
    )


    refined[
        excluded > 0
    ] = 0


    # Remove tiny noisy areas
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_OPEN,
        kernel
    )


    # Fill small gaps
    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        kernel
    )


    return refined

def feather_mask(
    mask,
    blur_size=21
):

    blurred = cv2.GaussianBlur(
        mask,
        (
            blur_size,
            blur_size
        ),
        0
    )

    return blurred
