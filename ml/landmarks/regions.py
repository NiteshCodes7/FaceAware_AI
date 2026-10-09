import mediapipe as mp


class FaceRegions:

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh

    # --------------------------------------------------
    # Convert MediaPipe connections into landmark indices
    # --------------------------------------------------

    @staticmethod
    def _connection_indices(connections):
        indices = set()

        for start, end in connections:
            indices.add(start)
            indices.add(end)

        return sorted(indices)

    # --------------------------------------------------
    # Eyes
    # --------------------------------------------------

    def get_left_eye(self):
        return self._connection_indices(
            self.face_mesh.FACEMESH_LEFT_EYE
        )

    def get_right_eye(self):
        return self._connection_indices(
            self.face_mesh.FACEMESH_RIGHT_EYE
        )

    # --------------------------------------------------
    # Iris
    # --------------------------------------------------

    def get_iris(self):
        return self._connection_indices(
            self.face_mesh.FACEMESH_IRISES
        )

    # --------------------------------------------------
    # Lips / mouth
    # --------------------------------------------------

    def get_mouth(self):
        return self._connection_indices(
            self.face_mesh.FACEMESH_LIPS
        )

    # --------------------------------------------------
    # Face contour
    # --------------------------------------------------

    def get_face_contour(self):
        return self._connection_indices(
            self.face_mesh.FACEMESH_FACE_OVAL
        )