import cv2

from detection.face_detector import FaceDetector


image = cv2.imread("test.jpeg")

detector = FaceDetector()

faces = detector.detect(image)

print("Faces detected:", len(faces))

for face in faces:

    x = face["x"]
    y = face["y"]
    w = face["width"]
    h = face["height"]

    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

cv2.imwrite("detected.jpg", image)

print("Saved detected.jpg")