import os
import cv2
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

NUM_IMAGES = 1000

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_IMAGE_DIR = BASE_DIR / "datasets" / "celebAMask" / "raw" / "CelebA-HQ-img"
RAW_MASK_DIR = BASE_DIR / "datasets" / "celebAMask" / "raw" / "CelebAMask-HQ-mask-anno"

OUTPUT_IMAGE_DIR = BASE_DIR / "datasets" / "celebAMask" / "images"
OUTPUT_MASK_DIR = BASE_DIR / "datasets" / "celebAMask" / "masks"


# ============================================================
# LABEL MAPPING
# ============================================================
#
# These IDs match the labels used by our FaceAware parser.
#
# 0  background
# 1  skin
# 2  nose
# 3  eye_glasses
# 4  left_eye
# 5  right_eye
# 6  left_brow
# 7  right_brow
# 8  left_ear
# 9  right_ear
# 10 mouth
# 11 upper_lip
# 12 lower_lip
# 13 hair
# 14 hat
# 15 earring
# 16 neck_l
# 17 neck
# 18 cloth
#

LABELS = {
    "skin": 1,
    "nose": 2,
    "eye_g": 3,
    "l_eye": 4,
    "r_eye": 5,
    "l_brow": 6,
    "r_brow": 7,
    "l_ear": 8,
    "r_ear": 9,
    "mouth": 10,
    "u_lip": 11,
    "l_lip": 12,
    "hair": 13,
    "hat": 14,
    "ear_r": 15,
    "neck_l": 16,
    "neck": 17,
    "cloth": 18,
}


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND SOURCE IMAGES
# ============================================================

image_files = sorted(RAW_IMAGE_DIR.glob("*.jpg"))

if not image_files:
    raise RuntimeError(
        f"No JPG images found in:\n{RAW_IMAGE_DIR}"
    )


print("=" * 60)
print("CelebAMask-HQ Dataset Preparation")
print("=" * 60)

print(f"Raw image directory : {RAW_IMAGE_DIR}")
print(f"Raw mask directory  : {RAW_MASK_DIR}")
print(f"Output images       : {OUTPUT_IMAGE_DIR}")
print(f"Output masks        : {OUTPUT_MASK_DIR}")
print(f"Images available    : {len(image_files)}")
print(f"Images requested    : {NUM_IMAGES}")
print()


# ============================================================
# PROCESS IMAGES
# ============================================================

processed = 0
skipped = 0

for image_path in image_files[:NUM_IMAGES]:

    image_id = image_path.stem.zfill(5)

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[WARNING] Could not read image: {image_path}")
        skipped += 1
        continue

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # Create empty multi-class mask
    # --------------------------------------------------------

    combined_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    found_any_mask = False

    # --------------------------------------------------------
    # Find each facial component
    # --------------------------------------------------------

    for component_name, class_id in LABELS.items():

        mask_found = False

        # Annotation folders are 0 ... 14
        for folder_id in range(15):

            component_path = (
                RAW_MASK_DIR
                / str(folder_id)
                / f"{image_id}_{component_name}.png"
            )

            if not component_path.exists():
                continue

            component_mask = cv2.imread(
                str(component_path),
                cv2.IMREAD_GRAYSCALE
            )

            if component_mask is None:
                continue

            # Resize only if necessary
            if component_mask.shape != (height, width):
                component_mask = cv2.resize(
                    component_mask,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                )

            # Wherever this component exists,
            # assign its class ID.
            combined_mask[component_mask > 0] = class_id

            found_any_mask = True
            mask_found = True

            break

    # --------------------------------------------------------
    # Make sure at least one mask exists
    # --------------------------------------------------------

    if not found_any_mask:
        print(
            f"[SKIP] No masks found for image {image_id}"
        )
        skipped += 1
        continue

    # --------------------------------------------------------
    # Save processed image
    # --------------------------------------------------------

    output_image_path = (
        OUTPUT_IMAGE_DIR / f"{image_id}.jpg"
    )

    output_mask_path = (
        OUTPUT_MASK_DIR / f"{image_id}.png"
    )

    cv2.imwrite(
        str(output_image_path),
        image
    )

    cv2.imwrite(
        str(output_mask_path),
        combined_mask
    )

    processed += 1

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if processed % 50 == 0:
        print(
            f"Processed: {processed}/{min(NUM_IMAGES, len(image_files))}"
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)

print(f"Processed : {processed}")
print(f"Skipped   : {skipped}")

print()
print("Output:")
print(f"Images → {OUTPUT_IMAGE_DIR}")
print(f"Masks  → {OUTPUT_MASK_DIR}")