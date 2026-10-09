import cv2

from segmentation.parser import FaceParser
from segmentation.region_extractor import RegionExtractor
from enhancement.wrinkles import WrinkleEnhancer


IMAGE_PATH = "test3.jpg"
MODEL_PATH = "../models/face_parser/resnet18.pt"


image = cv2.imread(IMAGE_PATH)

if image is None:
    raise RuntimeError(
        f"Could not load test image: {IMAGE_PATH}"
    )

print("Image loaded:", image.shape)

# -----------------------------
# Face parsing
# -----------------------------

print("Loading face parser...")

parser = FaceParser(MODEL_PATH)

print("Face parser loaded.")

print("Running segmentation...")

segmentation = parser.predict(image)

print("Segmentation completed.")


# -----------------------------
# Get skin region
# -----------------------------

regions = RegionExtractor(segmentation)

skin_mask = regions.get_skin()

cv2.imwrite(
    "wrinkle_skin_mask.png",
    skin_mask
)

print("Skin mask generated.")


# -----------------------------
# Wrinkle detection
# -----------------------------

enhancer = WrinkleEnhancer()

print("Detecting wrinkle-like regions...")

wrinkle_mask = enhancer.create_wrinkle_mask(
    image,
    skin_mask
)

cv2.imwrite(
    "wrinkle_mask.png",
    wrinkle_mask
)

print("Wrinkle mask generated.")


# -----------------------------
# Wrinkle reduction
# -----------------------------

for strength in [0.0, 0.5, 1.0]:

    result = enhancer.reduce(
        image,
        wrinkle_mask,
        strength=strength
    )

    percentage = int(strength * 100)

    cv2.imwrite(
        f"wrinkles_reduced_{percentage}.jpg",
        result
    )

    print(
        f"Generated wrinkles_reduced_{percentage}.jpg"
    )


print("\nWrinkle reduction test completed.")