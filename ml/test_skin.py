import cv2

from segmentation.parser import FaceParser
from segmentation.region_extractor import RegionExtractor

from enhancement.skin import smooth_skin


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


skin_mask = regions.get_skin()


result = smooth_skin(
    image,
    skin_mask,
    strength=0.5
)


cv2.imwrite(
    "skin_enhanced.jpg",
    result
)


print(
    "Skin enhancement completed."
)