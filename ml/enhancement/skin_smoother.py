import cv2


class SkinSmoother:

    def apply(self, image):
        """
        Generate a smoothed version of the image.

        This function does NOT perform masking or blending.
        """

        return cv2.bilateralFilter(
            image,
            d=9,
            sigmaColor=50,
            sigmaSpace=50
        )