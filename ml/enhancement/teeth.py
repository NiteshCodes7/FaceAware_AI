import cv2
import numpy as np


class TeethEnhancer:

    def whiten(self, image, teeth_mask, strength=0.5):
        """
        Whiten teeth using LAB color space.

        strength:
            0.0 -> original
            1.0 -> maximum enhancement
        """

        strength = float(np.clip(strength, 0.0, 1.0))

        if strength == 0:
            return image.copy()

        # --------------------------------
        # Convert BGR -> LAB
        # --------------------------------

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        L, A, B = cv2.split(lab)

        # --------------------------------
        # Brightness enhancement
        # --------------------------------

        L_float = L.astype(np.float32)

        L_enhanced = L_float + (25.0 * strength)

        L_enhanced = np.clip(
            L_enhanced,
            0,
            255
        ).astype(np.uint8)

        # --------------------------------
        # Reduce yellow tone
        #
        # In LAB, B represents the
        # blue <-> yellow component.
        # --------------------------------

        B_float = B.astype(np.float32)

        B_enhanced = B_float - (
            15.0 * strength
        )

        B_enhanced = np.clip(
            B_enhanced,
            0,
            255
        ).astype(np.uint8)

        # --------------------------------
        # Keep A mostly unchanged
        # --------------------------------

        enhanced_lab = cv2.merge(
            [
                L_enhanced,
                A,
                B_enhanced
            ]
        )

        enhanced = cv2.cvtColor(
            enhanced_lab,
            cv2.COLOR_LAB2BGR
        )

        # --------------------------------
        # Soft mask
        # --------------------------------

        soft_mask = cv2.GaussianBlur(
            teeth_mask,
            (7, 7),
            0
        )

        alpha = (
            soft_mask.astype(np.float32) / 255.0
        )

        alpha *= strength

        alpha = alpha[..., None]

        # --------------------------------
        # Blend
        # --------------------------------

        result = (
            image.astype(np.float32) * (1.0 - alpha)
            +
            enhanced.astype(np.float32) * alpha
        )

        return np.clip(
            result,
            0,
            255
        ).astype(np.uint8)