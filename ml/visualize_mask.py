import cv2
import numpy as np
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_PATH = Path("../datasets/celebAMask/images/00000.jpg")
MASK_PATH = Path("../datasets/celebAMask/masks/00000.png")

OUTPUT_PATH = Path("../datasets/celebAMask/mask_visualization.jpg")


# --------------------------------------------------
# Class colors
# --------------------------------------------------

COLORS = {
    0: (0, 0, 0),          # background
    1: (255, 180, 180),    # skin
    2: (0, 255, 255),      # nose
    3: (255, 0, 0),        # glasses
    4: (0, 255, 0),        # left eye
    5: (0, 200, 0),        # right eye
    6: (255, 255, 0),      # left eyebrow
    7: (200, 200, 0),      # right eyebrow
    8: (255, 0, 255),      # left ear
    9: (200, 0, 255),      # right ear
    10: (0, 128, 255),     # mouth
    11: (0, 0, 255),       # upper lip
    12: (100, 0, 255),     # lower lip
    13: (128, 64, 0),      # hair
    14: (0, 128, 128),     # hat
    15: (128, 0, 128),     # earring
    16: (64, 128, 128),    # neck_l
    17: (64, 64, 255),     # neck
    18: (128, 128, 128),   # cloth
}


# --------------------------------------------------
# Load image
# --------------------------------------------------

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")


# --------------------------------------------------
# Load mask
# --------------------------------------------------

mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise FileNotFoundError(f"Could not load mask: {MASK_PATH}")


# --------------------------------------------------
# Resize mask to image size
# --------------------------------------------------

mask = cv2.resize(
    mask,
    (image.shape[1], image.shape[0]),
    interpolation=cv2.INTER_NEAREST
)


# --------------------------------------------------
# Create colored mask
# --------------------------------------------------

colored_mask = np.zeros_like(image)

for class_id, color in COLORS.items():
    colored_mask[mask == class_id] = color


# --------------------------------------------------
# Create overlay
# --------------------------------------------------

overlay = cv2.addWeighted(
    image,
    0.55,
    colored_mask,
    0.45,
    0
)


# --------------------------------------------------
# Add class labels to console
# --------------------------------------------------

classes = np.unique(mask)

print("Classes present:")
for class_id in classes:
    print(f"  {class_id}")


# --------------------------------------------------
# Save visualization
# --------------------------------------------------

cv2.imwrite(str(OUTPUT_PATH), overlay)

print()
print(f"Visualization saved to:")
print(OUTPUT_PATH)