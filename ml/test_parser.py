import cv2
import numpy as np

from segmentation.parser import FaceParser
from segmentation.region_extractor import RegionExtractor
from segmentation.visualizer import create_overlay
from analysis.skin_analyzer import SkinAnalyzer
from enhancement.skin_smoother import SkinSmoother
from enhancement.engine import EnhancementEngine

IMAGE_PATH = "test3.jpg"
MODEL_PATH = "../models/face_parser/resnet18.pt"


# --------------------------------------------------
# 1. Load image
# --------------------------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise RuntimeError(
        f"Could not load test image: {IMAGE_PATH}"
    )

print("Image loaded:", image.shape)


# --------------------------------------------------
# 2. Load face parser
# --------------------------------------------------

print("Loading face parser...")

parser = FaceParser(MODEL_PATH)

print("Face parser loaded.")


# --------------------------------------------------
# 3. Run face segmentation
# --------------------------------------------------

print("Running segmentation...")

segmentation = parser.predict(image)

print("Segmentation shape:", segmentation.shape)


# --------------------------------------------------
# 4. Show detected classes
# --------------------------------------------------

unique_classes, counts = np.unique(
    segmentation,
    return_counts=True
)

print("\nDetected classes:")

for class_id, count in zip(unique_classes, counts):
    print(
        f"Class {class_id:2d}: "
        f"{count:7d} pixels"
    )


# --------------------------------------------------
# 5. Create visualization
# --------------------------------------------------

print("\nCreating segmentation overlay...")

overlay = create_overlay(
    image,
    segmentation
)

cv2.imwrite(
    "parser_overlay.png",
    overlay
)


# --------------------------------------------------
# 6. Extract regions
# --------------------------------------------------

regions = RegionExtractor(
    segmentation
)

skin = regions.get_skin()

skin_analyzer = SkinAnalyzer()

skin_analysis = skin_analyzer.analyze(
    image,
    skin
)

print("\nSkin Analysis:")
print(skin_analysis)

skin_smoother = SkinSmoother()

smoothed = skin_smoother.apply(image)

for strength in [0.0, 0.5, 1.0]:

    result = EnhancementEngine.blend(
        original=image,
        enhanced=smoothed,
        mask=skin,
        strength=strength
    )

    percentage = int(strength * 100)

    cv2.imwrite(
        f"skin_smoothed_{percentage}.png",
        result
    )

eyes = regions.get_eyes()
mouth = regions.get_mouth()
lips = regions.get_lips()
hair = regions.get_hair()
glasses = regions.get_glasses()


# --------------------------------------------------
# 7. Save individual masks
# --------------------------------------------------

cv2.imwrite(
    "parser_skin.png",
    skin
)

cv2.imwrite(
    "parser_eyes.png",
    eyes
)

cv2.imwrite(
    "parser_mouth.png",
    mouth
)

cv2.imwrite(
    "parser_lips.png",
    lips
)

cv2.imwrite(
    "parser_hair.png",
    hair
)

cv2.imwrite(
    "parser_glasses.png",
    glasses
)


print("\nParser test completed successfully.")

print("\nGenerated files:")
print("  parser_overlay.png")
print("  parser_skin.png")
print("  parser_eyes.png")
print("  parser_mouth.png")
print("  parser_lips.png")
print("  parser_hair.png")
print("  parser_glasses.png")
