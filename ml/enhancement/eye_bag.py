import cv2
import numpy as np


class EyeBagEnhancer:

    def _create_darkness_map(self, L):
        """
        Estimate how much darker each pixel is compared
        with its surrounding skin region.
        """

        # Large blur estimates the local skin brightness
        local_background = cv2.GaussianBlur(
            L,
            (0, 0),
            sigmaX=18,
            sigmaY=18
        )

        # Positive values = pixel is darker than surroundings
        darkness = (
            local_background.astype(np.float32)
            - L.astype(np.float32)
        )

        darkness = np.clip(darkness, 0, None)

        # Ignore very small natural variations
        darkness = np.clip(
            darkness - 3.0,
            0,
            None
        )

        # Normalize
        darkness = np.clip(
            darkness / 30.0,
            0.0,
            1.0
        )

        # Smooth the correction map
        darkness = cv2.GaussianBlur(
            darkness,
            (0, 0),
            sigmaX=5,
            sigmaY=5
        )

        return darkness

    def reduce(
        self,
        image,
        under_eye_mask,
        strength=0.5
    ):
        """
        Reduce dark eye bags using local brightness correction
        while preserving natural skin texture.

        strength:
            0.0 = original
            0.25 = subtle
            0.5 = moderate
            0.75 = strong
            1.0 = maximum
        """

        if image is None:
            raise ValueError("Image cannot be None")

        if under_eye_mask is None:
            raise ValueError(
                "Under-eye mask cannot be None"
            )

        if under_eye_mask.shape[:2] != image.shape[:2]:
            raise ValueError(
                "Image and under-eye mask dimensions must match"
            )

        strength = float(
            np.clip(strength, 0.0, 1.0)
        )

        if strength == 0:
            return image.copy()

        # --------------------------------------------------
        # 1. Convert image to LAB
        # --------------------------------------------------

        lab = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB
        )

        L, A, B = cv2.split(lab)

        # --------------------------------------------------
        # 2. Estimate local darkness
        # --------------------------------------------------

        darkness = self._create_darkness_map(L)

        # --------------------------------------------------
        # 3. Restrict correction to under-eye region
        # --------------------------------------------------

        region = (
            under_eye_mask.astype(np.float32) / 255.0
        )

        correction = darkness * region

        # --------------------------------------------------
        # 4. Lift darker areas
        # --------------------------------------------------

        # Maximum correction at strength 1
        # is approximately 18 LAB lightness units.
        brightness_gain = (
            correction * 18.0 * strength
        )

        L_enhanced = np.clip(
            L.astype(np.float32) + brightness_gain,
            0,
            255
        ).astype(np.uint8)

        # --------------------------------------------------
        # 5. Slightly reduce excessive dark color
        # --------------------------------------------------

        # Eye bags can sometimes appear slightly reddish/
        # bluish due to shadows. Pull chroma very gently
        # toward neutral only where darkness is detected.

        neutral_strength = (
            correction * 0.08 * strength
        )

        A_float = A.astype(np.float32)
        B_float = B.astype(np.float32)

        A_enhanced = (
            A_float
            + (128.0 - A_float) * neutral_strength
        )

        B_enhanced = (
            B_float
            + (128.0 - B_float) * neutral_strength
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

        enhanced_lab = cv2.merge([
            L_enhanced,
            A_enhanced,
            B_enhanced
        ])

        enhanced = cv2.cvtColor(
            enhanced_lab,
            cv2.COLOR_LAB2BGR
        )

        # --------------------------------------------------
        # 6. Preserve natural skin texture
        # --------------------------------------------------

        # Very light bilateral smoothing, only as a
        # secondary step. We don't want plastic skin.
        smooth = cv2.bilateralFilter(
            enhanced,
            d=5,
            sigmaColor=25,
            sigmaSpace=25
        )

        # --------------------------------------------------
        # 7. Soft final blending
        # --------------------------------------------------

        soft_mask = cv2.GaussianBlur(
            under_eye_mask,
            (21, 21),
            0
        )

        alpha = (
            soft_mask.astype(np.float32) / 255.0
        )

        alpha *= strength

        # Don't apply correction where there is
        # no detected darkness.
        alpha *= np.clip(
            darkness * 1.5,
            0.0,
            1.0
        )

        alpha = alpha[..., None]

        result = (
            image.astype(np.float32) * (1.0 - alpha)
            + smooth.astype(np.float32) * alpha
        )

        return np.clip(
            result,
            0,
            255
        ).astype(np.uint8)