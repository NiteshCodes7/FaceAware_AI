import numpy as np


class EnhancementEngine:

    @staticmethod
    def blend(original, enhanced, mask, strength=1.0):
        """
        Blend an enhanced image into the original
        using a region mask.

        strength:
            0.0 = original
            1.0 = full enhancement
        """

        if not 0.0 <= strength <= 1.0:
            raise ValueError(
                "strength must be between 0.0 and 1.0"
            )

        # Convert mask from [0, 255] to [0, 1]
        alpha = mask.astype(np.float32) / 255.0

        # Apply user-controlled strength
        alpha *= strength

        # H x W -> H x W x 1
        alpha = np.expand_dims(alpha, axis=2)

        original_float = original.astype(np.float32)
        enhanced_float = enhanced.astype(np.float32)

        result = (
            original_float * (1.0 - alpha)
            + enhanced_float * alpha
        )

        return np.clip(
            result,
            0,
            255
        ).astype(np.uint8)