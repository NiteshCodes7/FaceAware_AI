import cv2
import numpy as np

from segmentation.parser import FaceParser
from segmentation.region_extractor import RegionExtractor


image = cv2.imread(
    "test.jpeg"
)

parser = FaceParser(
    "models/face_parser/"
    "unet_face_parser.pth"
)

segmentation = parser.predict(
    image
)

regions = RegionExtractor(
    segmentation
)


skin = regions.get_skin()
eyes = regions.get_eyes()
mouth = regions.get_mouth()
hair = regions.get_hair()


overlay = image.copy()


# Skin
overlay[skin > 0] = (
    0,
    255,
    0
)


# Eyes
overlay[eyes > 0] = (
    255,
    0,
    0
)


# Mouth
overlay[mouth > 0] = (
    0,
    0,
    255
)


# Hair
overlay[hair > 0] = (
    255,
    255,
    0
)


result = cv2.addWeighted(
    image,
    0.65,
    overlay,
    0.35,
    0
)


cv2.imwrite(
    "parser_visualization.jpg",
    result
)


print(
    "Saved parser_visualization.jpg"
)