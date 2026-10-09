import cv2
import numpy as np


class LandmarkMaskGenerator:

    def __init__(self, image_shape):
        """
        image_shape:
            image.shape
        """
        self.height = image_shape[0]
        self.width = image_shape[1]

    def polygon_mask(self, landmarks, indices):
        """
        Create a filled polygon mask from landmark indices.
        """

        mask = np.zeros(
            (self.height, self.width),
            dtype=np.uint8
        )

        points = np.array(
            [landmarks[i] for i in indices],
            dtype=np.int32
        )

        if len(points) >= 3:
            cv2.fillPoly(
                mask,
                [points],
                255
            )

        return mask

    def eye_mask(self, landmarks, indices):
        """
        Create eye mask and slightly expand it.
        """

        mask = self.polygon_mask(
            landmarks,
            indices
        )

        kernel = np.ones(
            (3, 3),
            dtype=np.uint8
        )

        mask = cv2.dilate(
            mask,
            kernel,
            iterations=1
        )

        return mask

    def iris_mask(self, landmarks, indices):
        """
        Create precise iris mask.
        """

        return self.polygon_mask(
            landmarks,
            indices
        )

    def mouth_mask(self, landmarks, indices):
        """
        Create mouth region mask.
        """

        return self.polygon_mask(
            landmarks,
            indices
        )

    def create_all_masks(
        self,
        landmarks,
        left_eye_indices,
        right_eye_indices,
        left_iris_indices,
        right_iris_indices,
        mouth_indices
    ):
        """
        Generate landmark-based facial masks.
        """

        return {
            "left_eye": self.eye_mask(
                landmarks,
                left_eye_indices
            ),

            "right_eye": self.eye_mask(
                landmarks,
                right_eye_indices
            ),

            "left_iris": self.iris_mask(
                landmarks,
                left_iris_indices
            ),

            "right_iris": self.iris_mask(
                landmarks,
                right_iris_indices
            ),

            "mouth": self.mouth_mask(
                landmarks,
                mouth_indices
            )
        }