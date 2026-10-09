import cv2


def draw_landmarks(image, face_landmarks):

    output = image.copy()

    height, width = image.shape[:2]

    for landmark in face_landmarks.landmark:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        # Ignore points outside the image
        if 0 <= x < width and 0 <= y < height:

            cv2.circle(
                output,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )

    return output