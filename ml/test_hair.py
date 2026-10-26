import cv2

from segmentation.parser import FaceParser
from enhancement.hair import HairEnhancer

image = cv2.imread("1.jpg")

parser = FaceParser("../models/face_parser/training/bisenet_celebamask_best.pt")
segmentation_mask = parser.predict(image)

enhancer = HairEnhancer(gamma = 1.4, darkening_strength = 0.9)
result = enhancer.enhance(image, segmentation_mask, strength = 0.75)

cv2.imwrite("hair_original.jpg", image)
cv2.imwrite("hair_enhanced.jpg", result)

# Save the hair mask to verify the correct region is selected.
hair_mask = (
    segmentation_mask == 13
).astype("uint8") * 255

cv2.imwrite("hair_mask.jpg", hair_mask)

print("Saved hair enhancement results.")