import cv2
import numpy as np


class RegionMasks:

    def __init__(self, image_shape):

        self.height = image_shape[0]
        self.width = image_shape[1]

    def polygon_mask(self, points):

        mask = np.zeros(
            (self.height, self.width),
            dtype=np.uint8
        )

        points = np.array(points, dtype=np.int32)

        cv2.fillPoly(
            mask,
            [points],
            255
        )

        return mask

    def create_eye_mask(self, landmarks, indices):

        points = [
            landmarks[i]
            for i in indices
        ]

        return self.polygon_mask(points)

    def create_mouth_mask(self, landmarks, indices):

        points = [
            landmarks[i]
            for i in indices
        ]

        return self.polygon_mask(points)

    def create_inner_mouth_mask(self, landmarks, indices):

        points = [
            landmarks[i]
            for i in indices
        ]

        return self.polygon_mask(points)

    def create_iris_mask(self, landmarks, indices):
        """
        Create a smooth elliptical iris mask from
        the four MediaPipe iris boundary landmarks.
        """

        points = np.array(
            [landmarks[i] for i in indices],
            dtype=np.float32
        )

        if len(points) < 4:
            return np.zeros(
                (self.height, self.width),
                dtype=np.uint8
            )

        # --------------------------------
        # Estimate iris center
        # --------------------------------

        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])

        center = (
            int(round(center_x)),
            int(round(center_y))
        )

        # --------------------------------
        # Estimate iris dimensions
        # --------------------------------

        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])

        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])

        radius_x = (max_x - min_x) / 2.0
        radius_y = (max_y - min_y) / 2.0

        # Slight expansion so the complete visible
        # iris is covered.
        radius_x *= 1.15
        radius_y *= 1.15

        radius_x = max(1, int(round(radius_x)))
        radius_y = max(1, int(round(radius_y)))

        # --------------------------------
        # Create elliptical mask
        # --------------------------------

        mask = np.zeros(
            (self.height, self.width),
            dtype=np.uint8
        )

        cv2.ellipse(
            mask,
            center,
            (radius_x, radius_y),
            0,
            0,
            360,
            255,
            -1
        )

        # --------------------------------
        # Slight smoothing
        # --------------------------------

        mask = cv2.GaussianBlur(
            mask,
            (3, 3),
            0
        )

        return mask