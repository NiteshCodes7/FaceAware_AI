import cv2
import numpy as np


class EyeBagMask:

    def __init__(self, image_shape):
        self.height = image_shape[0]
        self.width = image_shape[1]

    def create(self, landmarks, eye_indices):
        """
        Create an under-eye region from eye landmarks.
        """

        eye_points = np.array(
            [landmarks[i] for i in eye_indices],
            dtype=np.float32
        )

        min_x = int(np.min(eye_points[:, 0]))
        max_x = int(np.max(eye_points[:, 0]))

        min_y = int(np.min(eye_points[:, 1]))
        max_y = int(np.max(eye_points[:, 1]))

        eye_height = max_y - min_y

        # Region below the eye
        padding_x = int((max_x - min_x) * 0.10)

        start_x = max(0, min_x - padding_x)
        end_x = min(self.width, max_x + padding_x)

        start_y = min(
            self.height,
            max_y + int(eye_height * 0.20)
        )

        end_y = min(
            self.height,
            max_y + int(eye_height * 1.8)
        )

        mask = np.zeros(
            (self.height, self.width),
            dtype=np.uint8
        )

        # Ellipse gives a much more natural region
        center_x = (start_x + end_x) // 2
        center_y = (start_y + end_y) // 2

        axis_x = max(1, (end_x - start_x) // 2)
        axis_y = max(1, (end_y - start_y) // 2)

        cv2.ellipse(
            mask,
            (center_x, center_y),
            (axis_x, axis_y),
            0,
            0,
            360,
            255,
            -1
        )

        # Soft boundary
        mask = cv2.GaussianBlur(
            mask,
            (15, 15),
            0
        )

        return mask