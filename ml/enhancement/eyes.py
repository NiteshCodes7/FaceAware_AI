import cv2
import numpy as np


class EyeEnhancer:

    def __init__(self):
        pass

    def create_sclera_mask(
        self,
        eye_mask,
        iris_mask
    ):
        """
        Remove the iris region from the eye mask.

        Sclera = Eye - Iris
        """

        # Make sure both masks are binary
        eye_binary = np.where(
            eye_mask > 0,
            255,
            0
        ).astype(np.uint8)

        iris_binary = np.where(
            iris_mask > 0,
            255,
            0
        ).astype(np.uint8)

        # Remove iris from eye
        sclera_mask = cv2.bitwise_and(
            eye_binary,
            cv2.bitwise_not(iris_binary)
        )

        return sclera_mask

    def whiten(
        self,
        image,
        sclera_mask,
        strength=0.5
    ):
        """
        Whiten the sclera region.

        strength:
            0.0 -> original
            1.0 -> maximum enhancement
        """

        strength = float(
            np.clip(
                strength,
                0.0,
                1.0
            )
        )

        if strength == 0:
            return image.copy()

        # --------------------------------
        # Convert image to LAB
        # --------------------------------

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )

        L, A, B = cv2.split(lab)

        # --------------------------------
        # Brightness enhancement
        # --------------------------------

        L_float = L.astype(
            np.float32
        )

        # Increase luminance gradually
        L_enhanced = L_float + (
            20.0 * strength
        )

        L_enhanced = np.clip(
            L_enhanced,
            0,
            255
        ).astype(np.uint8)

        # --------------------------------
        # Slightly reduce yellow/red tint
        # --------------------------------

        A_float = A.astype(
            np.float32
        )

        B_float = B.astype(
            np.float32
        )

        # Move A and B slightly toward
        # neutral depending on strength.
        A_enhanced = (
            A_float +
            (128.0 - A_float) * 0.15 * strength
        )

        B_enhanced = (
            B_float +
            (128.0 - B_float) * 0.20 * strength
        )

        A_enhanced = np.clip(
            A_enhanced,
            0,
            255
        ).astype(np.uint8)

        B_enhanced = np.clip(
            B_enhanced,
            0,
            255
        ).astype(np.uint8)

        # --------------------------------
        # Reconstruct LAB image
        # --------------------------------

        enhanced_lab = cv2.merge(
            [
                L_enhanced,
                A_enhanced,
                B_enhanced
            ]
        )

        enhanced = cv2.cvtColor(
            enhanced_lab,
            cv2.COLOR_LAB2BGR
        )

        # --------------------------------
        # Feather the mask
        # --------------------------------

        soft_mask = cv2.GaussianBlur(
            sclera_mask,
            (7, 7),
            0
        )

        alpha = (
            soft_mask.astype(np.float32)
            / 255.0
        )

        # Apply requested strength
        alpha *= strength

        alpha = alpha[..., None]

        # --------------------------------
        # Blend
        # --------------------------------

        result = (
            image.astype(np.float32)
            * (1.0 - alpha)
            +
            enhanced.astype(np.float32)
            * alpha
        )

        return np.clip(
            result,
            0,
            255
        ).astype(np.uint8)