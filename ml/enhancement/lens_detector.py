import cv2
import numpy as np


class LensDetector:
    """
    Builds a lens mask from MediaPipe landmarks (needs refine_landmarks=True,
    i.e. 478 points incl. iris).
    Lens centre = iris centre + (dx, dy) * IPD, rotated with head roll.
    """

    def __init__(self, dx=-0.11, dy=0.13, radius=0.42, aspect=1.0):
        self.dx = dx
        self.dy = dy
        self.radius = radius
        self.aspect = aspect

    def _build(self, left, right, shape):
        h, w = shape[:2]

        v = right - left
        ipd = float(np.linalg.norm(v))
        if ipd < 1:
            raise RuntimeError("Invalid eye distance")

        roll = float(np.arctan2(v[1], v[0]))
        c, s = np.cos(roll), np.sin(roll)

        off = np.array([
            self.dx * ipd * c - self.dy * ipd * s,
            self.dx * ipd * s + self.dy * ipd * c,
        ])

        axes = (
            max(1, int(self.radius * ipd)),
            max(1, int(self.radius * ipd * self.aspect)),
        )
        angle = float(np.degrees(roll))

        mask = np.zeros((h, w), np.uint8)
        for eye in (left, right):
            cx, cy = (eye + off).astype(int)
            cv2.ellipse(mask, (int(cx), int(cy)), axes, angle, 0, 360, 255, -1)
        return mask

    def detect_from_points(self, points, shape):
        """points: (478, 2+) array/list, pixel coords (normalized 0-1 also accepted)."""
        h, w = shape[:2]
        pts = np.asarray(points, dtype=np.float32)[:, :2]

        if len(pts) < 478:
            raise RuntimeError("Need 478 landmarks (refine_landmarks=True)")

        if pts.max() <= 1.5:                      # normalized -> pixels
            pts = pts * np.array([w, h], np.float32)

        a = pts[468:473].mean(axis=0)
        b = pts[473:478].mean(axis=0)
        left, right = (a, b) if a[0] < b[0] else (b, a)
        return self._build(left, right, shape)

    def detect(self, image):
        """Standalone use: runs its own FaceMesh."""
        import mediapipe as mp

        h, w = image.shape[:2]
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True
        ) as fm:
            res = fm.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not res.multi_face_landmarks:
            raise RuntimeError("No face detected")

        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([[p.x * w, p.y * h] for p in lm], np.float32)
        return self.detect_from_points(pts, image.shape)