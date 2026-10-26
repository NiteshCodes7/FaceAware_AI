import cv2
import numpy as np


class HairEnhancer:
    """
    Enhances hair contrast and texture using the BiSeNet
    segmentation mask.

    FaceAware AI class mapping:
        13 = hair
    """

    GAMMA = 1.40
    DARKENINIG_STRENGTH = 0.90

    HAIR_CLASS = 13

    def __init__(self, gamma = GAMMA, darkening_strength = DARKENINIG_STRENGTH):
        if gamma <= 0:
            raise ValueError("Gamma must be greater than 0.")

        if not 0.0 <= darkening_strength <= 1.0:
            raise ValueError(
                "Darkening strength must be between 0 and 1."
            )

        self.gamma = float(gamma)
        self.darkening_strength = float(darkening_strength)

    def enhance(self, image, segmentation_mask, strength = 0.5):
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")

        if segmentation_mask is None:
            raise ValueError("Segmentation mask is missing.")

        self.strength = float(np.clip(strength, 0.0, 1.0))

        if segmentation_mask.shape[:2] != image.shape[:2]:
            segmentation_mask = cv2.resize(
                segmentation_mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Extract only the hair pixels.
        hair_mask = (
            segmentation_mask == self.HAIR_CLASS
        ).astype(np.uint8) * 255

        if cv2.countNonZero(hair_mask) == 0:
            return image.copy()

        # Slightly expand the mask to cover hair boundaries.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)
        )
        hair_mask = cv2.dilate(hair_mask, kernel, iterations = 1)

        # Smooth mask edges to avoid visible outlines.
        alpha = cv2.GaussianBlur(
            hair_mask.astype(np.float32) / 255.0,
            (0, 0),
            sigmaX=2,
        )
        alpha = np.clip(alpha * self.strength, 0.0, 1.0)
        alpha = alpha[..., None]

        # Work in LAB color space to darken hair naturally.
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)

        lightness_float = lightness.astype(np.float32)

        # Darken the hair while preserving strand details.
        # Gamma > 1 reduces brightness, especially in lighter hair regions.
        

        darkened_lightness = 255.0 * np.power(
            lightness_float / 255.0,
            self.gamma
        )

        # Blend the darkened lightness with the original.
        # This preserves texture and avoids flat, artificial black hair.
        

        enhanced_lightness = (
            (1.0 - self.darkening_strength) * lightness_float
            + self.darkening_strength * darkened_lightness
        )

        enhanced_lightness = np.clip(
            enhanced_lightness, 0, 255
        ).astype(np.uint8)

        # Preserve the original color channels.
        enhanced_lab = cv2.merge([
            enhanced_lightness,
            a_channel,
            b_channel
        ])

        enhanced = cv2.cvtColor(
            enhanced_lab, cv2.COLOR_LAB2BGR
        )


        # Emphasize fine hair details.
        blurred = cv2.GaussianBlur(
            enhanced, (0, 0), sigmaX=1.0
        )

        sharpened = cv2.addWeighted(
            enhanced, 2.0, blurred, -1.0, 0
        )

        # Blend the result only inside the hair region.
        result = (
            image.astype(np.float32) * (1.0 - alpha)
            + sharpened.astype(np.float32) * alpha
        )

        result = np.clip(result, 0, 255).astype(np.uint8)

        return result
