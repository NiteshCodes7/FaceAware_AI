import cv2
import os

from landmarks.face_landmarks import FaceLandmarks
from landmarks.visualizer import draw_landmarks


IMAGE_PATH = "test.jpeg"


print("Starting landmark test...")


# --------------------------------------------------
# Load image
# --------------------------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise RuntimeError(
        f"Could not load image: {IMAGE_PATH}"
    )

print("Image loaded:", image.shape)


# --------------------------------------------------
# Detect landmarks
# --------------------------------------------------

detector = FaceLandmarks()

print("Running face detection...")

faces = detector.detect(image)

print("Faces detected:", len(faces))


# --------------------------------------------------
# Draw landmarks
# --------------------------------------------------

if len(faces) == 0:

    print("ERROR: No face detected.")

else:

    print("Face detected!")

    print("Drawing landmarks...")

    landmark_image = draw_landmarks(
        image,
        faces[0]
    )

    print("Saving image...")

    output_path = os.path.abspath(
        "face_landmarks.png"
    )

    success = cv2.imwrite(
        output_path,
        landmark_image
    )

    print(
        "cv2.imwrite result:",
        success
    )

    print(
        "Output path:",
        output_path
    )


print("Test finished.")