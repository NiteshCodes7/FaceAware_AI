import cv2

from landmarks.face_landmarks import FaceLandmarks
from landmarks.visualizer import draw_landmarks


IMAGE_PATH = "test.jpeg"


print("Starting landmark test...")


# Load image
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise RuntimeError(
        f"Could not load image: {IMAGE_PATH}"
    )

print("Image loaded:", image.shape)


# Create detector
detector = FaceLandmarks()

print("Running face detection...")

faces = detector.detect(image)

print("Faces detected:", len(faces))


if len(faces) == 0:

    print("ERROR: No face detected.")

else:

    print("Face detected!")

    # Draw landmarks
    print("Drawing landmarks...")

    landmark_image = draw_landmarks(
        image,
        faces[0]
    )

    print("Saving image...")

    success = cv2.imwrite(
        "face_landmarks.png",
        landmark_image
    )

    print("cv2.imwrite result:", success)

    print(
        "Absolute output path:"
    )

    import os
    print(
        os.path.abspath("face_landmarks.png")
    )


print("Test finished.")