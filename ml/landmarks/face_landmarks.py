import cv2
import mediapipe as mp


class FaceLandmarks:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def detect(self, image):
        """
        Detect faces and return a list of
        NormalizedLandmarkList objects.
        """

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = self.mesh.process(rgb)

        if not results.multi_face_landmarks:
            return []

        return results.multi_face_landmarks

    def get_points(self, image):
        """
        Get pixel coordinates for the first detected face.
        """

        faces = self.detect(image)

        if not faces:
            return None

        landmarks = faces[0]

        height, width = image.shape[:2]

        points = []

        for landmark in landmarks.landmark:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points.append((x, y))

        return points