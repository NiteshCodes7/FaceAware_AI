import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.bisenet import BiSeNet


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 19
IMAGE_SIZE = 512

BATCH_SIZE = 2
EPOCHS = 3

LEARNING_RATE = 1e-4

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATASET_ROOT = Path("../datasets/celebAMask/split")

PRETRAINED_MODEL = Path(
    "../models/face_parser/training/"
    "bisenet_celebamask_best.pt"
)

OUTPUT_DIR = Path(
    "../models/face_parser/training"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Dataset
# ============================================================

class CelebAMaskDataset(Dataset):

    def __init__(self, root, split):

        self.image_dir = root / split / "images"
        self.mask_dir = root / split / "masks"

        self.images = sorted(
            self.image_dir.glob("*.jpg")
        )

        if len(self.images) == 0:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]

        mask_path = (
            self.mask_dir /
            f"{image_path.stem}.png"
        )

        # --------------------------------------------
        # Load image
        # --------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # --------------------------------------------
        # Load mask
        # --------------------------------------------

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise RuntimeError(
                f"Could not read mask: {mask_path}"
            )

        # --------------------------------------------
        # Resize
        # --------------------------------------------

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        # --------------------------------------------
        # Convert image to tensor
        # --------------------------------------------

        image = torch.from_numpy(
            image
        ).permute(2, 0, 1).float() / 255.0

        # --------------------------------------------
        # Normalize image
        # --------------------------------------------

        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

        image = (
            image - mean
        ) / std

        # --------------------------------------------
        # Mask → Long tensor
        # --------------------------------------------

        mask = torch.from_numpy(
            mask.astype(np.int64)
        )

        return image, mask


# ============================================================
# Metrics
# ============================================================

def calculate_pixel_accuracy(pred, target):

    correct = (
        pred == target
    ).sum()

    total = target.numel()

    return (
        correct.float() / total
    ).item()


def calculate_miou(pred, target, num_classes):

    ious = []

    for class_id in range(num_classes):

        pred_class = pred == class_id
        target_class = target == class_id

        intersection = (
            pred_class & target_class
        ).sum().float()

        union = (
            pred_class | target_class
        ).sum().float()

        if union == 0:
            continue

        iou = (
            intersection / union
        ).item()

        ious.append(iou)

    if not ious:
        return 0.0

    return sum(ious) / len(ious)


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    total_miou = 0.0

    batches = 0

    for images, masks in loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        output = model(images)[0]

        loss = criterion(
            output,
            masks
        )

        predictions = torch.argmax(
            output,
            dim=1
        )

        accuracy = calculate_pixel_accuracy(
            predictions,
            masks
        )

        miou = calculate_miou(
            predictions,
            masks,
            NUM_CLASSES
        )

        total_loss += loss.item()
        total_accuracy += accuracy
        total_miou += miou

        batches += 1

    return (
        total_loss / batches,
        total_accuracy / batches,
        total_miou / batches
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("FaceAware AI - BiSeNet Training")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = CelebAMaskDataset(
        DATASET_ROOT,
        "train"
    )

    val_dataset = CelebAMaskDataset(
        DATASET_ROOT,
        "val"
    )

    print()
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Loading BiSeNet...")

    model = BiSeNet(
        num_classes=NUM_CLASSES,
        backbone_name="resnet18"
    )

    # --------------------------------------------------------
    # Load existing model
    # --------------------------------------------------------

    if PRETRAINED_MODEL.exists():

        print(
            f"Loading pretrained weights: "
            f"{PRETRAINED_MODEL}"
        )

        checkpoint = torch.load(
            PRETRAINED_MODEL,
            map_location=DEVICE
        )

        model.load_state_dict(
            checkpoint
        )

    else:

        print(
            "WARNING: pretrained model not found."
        )

        print(
            "Training from randomly initialized weights."
        )

    model.to(DEVICE)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_miou = 0.7226

    for epoch in range(EPOCHS):

        model.train()

        epoch_loss = 0.0

        start_time = time.time()

        for batch_index, (images, masks) in enumerate(
            train_loader
        ):

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # --------------------------------------------
            # Forward pass
            # --------------------------------------------

            optimizer.zero_grad()

            output = model(images)[0]

            # --------------------------------------------
            # Loss
            # --------------------------------------------

            loss = criterion(
                output,
                masks
            )

            # --------------------------------------------
            # Backpropagation
            # --------------------------------------------

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            # --------------------------------------------
            # Progress
            # --------------------------------------------

            if (
                batch_index + 1
            ) % 50 == 0:

                print(
                    f"Epoch "
                    f"{epoch + 1}/{EPOCHS} | "
                    f"Batch "
                    f"{batch_index + 1}/"
                    f"{len(train_loader)} | "
                    f"Loss: "
                    f"{loss.item():.4f}"
                )

        # ------------------------------------------------
        # Validation
        # ------------------------------------------------

        validation_loss, accuracy, miou = validate(
            model,
            val_loader,
            criterion
        )

        epoch_loss /= len(train_loader)

        elapsed = (
            time.time() - start_time
        )

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS} finished"
        )

        print(
            f"Train Loss: {epoch_loss:.4f}"
        )

        print(
            f"Val Loss: {validation_loss:.4f}"
        )

        print(
            f"Pixel Accuracy: {accuracy:.4f}"
        )

        print(
            f"mIoU: {miou:.4f}"
        )

        print(
            f"Time: {elapsed:.1f}s"
        )

        print("-" * 60)

        # ------------------------------------------------
        # Save best model
        # ------------------------------------------------

        if miou > best_miou:

            best_miou = miou

            best_path = (
                OUTPUT_DIR /
                "bisenet_celebamask_best.pt"
            )

            torch.save(
                model.state_dict(),
                best_path
            )

            print(
                f"New best model saved:"
            )

            print(best_path)

    print()
    print("=" * 60)
    print("Training completed.")
    print(f"Best validation mIoU: {best_miou:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()