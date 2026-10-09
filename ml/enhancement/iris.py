import cv2
import numpy as np


class IrisEnhancer:

    def __init__(self):
        pass

    def enhance(self, image, iris_mask, strength=0.5):

        strength = float(np.clip(strength, 0.0, 1.0))

        if strength == 0:
            return image.copy()

        # --------------------------------
        # Convert to HSV
        # --------------------------------

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        H, S, V = cv2.split(hsv)

        # --------------------------------
        # Increase iris contrast
        # --------------------------------

        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(4, 4)
        )

        V_enhanced = clahe.apply(V)

        # --------------------------------
        # Increase iris saturation
        # --------------------------------

        S_float = S.astype(np.float32)

        S_enhanced = S_float * (
            1.0 + 0.30 * strength
        )

        S_enhanced = np.clip(
            S_enhanced,
            0,
            255
        ).astype(np.uint8)

        # --------------------------------
        # Blend original and enhanced
        # --------------------------------

        V_blended = cv2.addWeighted(
            V,
            1.0 - strength,
            V_enhanced,
            strength,
            0
        )

        S_blended = cv2.addWeighted(
            S,
            1.0 - strength,
            S_enhanced,
            strength,
            0
        )

        # --------------------------------
        # Reconstruct HSV image
        # --------------------------------

        enhanced_hsv = cv2.merge(
            [
                H,
                S_blended,
                V_blended
            ]
        )

        enhanced = cv2.cvtColor(
            enhanced_hsv,
            cv2.COLOR_HSV2BGR
        )

        # --------------------------------
        # Soft iris mask
        # --------------------------------

        soft_mask = cv2.GaussianBlur(
            iris_mask,
            (7, 7),
            0
        )

        alpha = (
            soft_mask.astype(np.float32) / 255.0
        )

        alpha *= strength

        alpha = alpha[..., None]

        # --------------------------------
        # Final blending
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