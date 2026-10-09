import cv2 
import mediapipe as mp

class FaceDetector:

    def __init__(self):

        self.mp_face_detection = mp.solutions.face_detection

        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

    def detect(self, image):

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.detector.process(rgb)

        faces = []

        if results.detections:

            h, w, _ = image.shape

            for detection in results.detections:

                box = detection.location_data.relative_bounding_box

                x = int(box.xmin * w)
                y = int(box.ymin * h)

                width = int(box.width * w)
                height = int(box.height * h)

                x = max(0, x)
                y = max(0, y)

                width = min(width, w - x)
                height = min(height, h - y)

                faces.append(
                    {
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height
                    }
                )

        return faces