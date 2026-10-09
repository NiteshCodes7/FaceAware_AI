import cv2
import numpy as np


class WrinkleEnhancer:

    def create_wrinkle_mask(self, image, skin_mask):
        """
        Detect wrinkle-like dark structures inside the skin region.
        Uses multi-scale black-hat morphology.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        skin = skin_mask > 0

        responses = []

        for size in [7, 11, 15]:

            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (size, size)
            )

            blackhat = cv2.morphologyEx(
                gray,
                cv2.MORPH_BLACKHAT,
                kernel
            )

            responses.append(
                blackhat.astype(np.float32)
            )

        response = np.maximum.reduce(responses)

        skin_response = response[skin]

        if len(skin_response) == 0:
            return np.zeros_like(gray)

        nonzero = skin_response[
            skin_response > 0
        ]

        if len(nonzero) == 0:
            return np.zeros_like(gray)

        threshold = np.percentile(
            nonzero,
            75
        )

        candidate = (
            skin &
            (response >= threshold)
        ).astype(np.uint8) * 255

        small_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_OPEN,
            small_kernel,
            iterations=1
        )

        candidate = cv2.morphologyEx(
            candidate,
            cv2.MORPH_CLOSE,
            small_kernel,
            iterations=1
        )

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                candidate,
                connectivity=8
            )
        )

        wrinkle_mask = np.zeros_like(candidate)

        for i in range(1, num_labels):

            area = stats[
                i,
                cv2.CC_STAT_AREA
            ]

            if area >= 8:
                wrinkle_mask[
                    labels == i
                ] = 255

        return wrinkle_mask


    def reduce(
        self,
        image,
        wrinkle_mask,
        strength=0.5
    ):
        """
        Reduce wrinkles using targeted image inpainting.

        strength:
            0.0 -> original
            0.5 -> moderate reduction
            1.0 -> maximum reduction
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

        # ------------------------------------
        # Prepare wrinkle mask
        # ------------------------------------

        mask = wrinkle_mask.copy()

        # Slightly expand the detected wrinkle
        # so its edges are also reconstructed.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        mask = cv2.dilate(
            mask,
            kernel,
            iterations=1
        )

        # ------------------------------------
        # Targeted inpainting
        # ------------------------------------

        restored = cv2.inpaint(
            image,
            mask,
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA
        )

        # ------------------------------------
        # Soft transition
        # ------------------------------------

        soft_mask = cv2.GaussianBlur(
            mask,
            (7, 7),
            0
        )

        alpha = (
            soft_mask.astype(np.float32)
            / 255.0
        )

        # User-controlled strength
        alpha *= strength

        alpha = alpha[..., None]

        # ------------------------------------
        # Blend
        # ------------------------------------

        result = (
            image.astype(np.float32)
            * (1.0 - alpha)
            +
            restored.astype(np.float32)
            * alpha
        )

        return np.clip(
            result,
            0,
            255
        ).astype(np.uint8)