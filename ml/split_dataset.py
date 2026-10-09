from pathlib import Path
import random
import shutil


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SOURCE_IMAGES = Path("../datasets/celebAMask/images")
SOURCE_MASKS = Path("../datasets/celebAMask/masks")

OUTPUT_ROOT = Path("../datasets/celebAMask/split")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42


# --------------------------------------------------
# Collect image files
# --------------------------------------------------

images = sorted(SOURCE_IMAGES.glob("*.jpg"))

print(f"Total images found: {len(images)}")

if len(images) == 0:
    raise RuntimeError("No images found.")


# --------------------------------------------------
# Verify corresponding masks
# --------------------------------------------------

valid_images = []

for image_path in images:
    mask_path = SOURCE_MASKS / f"{image_path.stem}.png"

    if mask_path.exists():
        valid_images.append(image_path)
    else:
        print(f"Missing mask: {mask_path}")


images = valid_images

print(f"Valid image-mask pairs: {len(images)}")


# --------------------------------------------------
# Shuffle
# --------------------------------------------------

random.seed(SEED)
random.shuffle(images)


# --------------------------------------------------
# Calculate split sizes
# --------------------------------------------------

total = len(images)

train_count = int(total * TRAIN_RATIO)
val_count = int(total * VAL_RATIO)

train_images = images[:train_count]
val_images = images[train_count:train_count + val_count]
test_images = images[train_count + val_count:]


# --------------------------------------------------
# Create directories
# --------------------------------------------------

for split in ["train", "val", "test"]:
    (OUTPUT_ROOT / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_ROOT / split / "masks").mkdir(
        parents=True,
        exist_ok=True
    )


# --------------------------------------------------
# Copy files
# --------------------------------------------------

def copy_split(files, split):

    for image_path in files:

        mask_path = SOURCE_MASKS / f"{image_path.stem}.png"

        destination_image = (
            OUTPUT_ROOT / split / "images" / image_path.name
        )

        destination_mask = (
            OUTPUT_ROOT / split / "masks" / mask_path.name
        )

        shutil.copy2(image_path, destination_image)
        shutil.copy2(mask_path, destination_mask)


copy_split(train_images, "train")
copy_split(val_images, "val")
copy_split(test_images, "test")


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()
print("Dataset split completed.")
print("--------------------------------")
print(f"Train: {len(train_images)}")
print(f"Validation: {len(val_images)}")
print(f"Test: {len(test_images)}")
print("--------------------------------")
print(f"Output: {OUTPUT_ROOT}")