import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from models.bisenet import BiSeNet


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 19
IMAGE_SIZE = 512
BATCH_SIZE = 2
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATASET_ROOT = Path("../datasets/celebAMask/split")

MODEL_PATH = Path(
    "../models/face_parser/training/"
    "bisenet_celebamask_best.pt"
)


CLASS_NAMES = [
    "background",
    "skin",
    "nose",
    "glasses",
    "left_eye",
    "right_eye",
    "left_brow",
    "right_brow",
    "left_ear",
    "right_ear",
    "mouth",
    "upper_lip",
    "lower_lip",
    "hair",
    "hat",
    "earring",
    "neck_l",
    "neck",
    "cloth",
]


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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]

        mask_path = (
            self.mask_dir /
            f"{image_path.stem}.png"
        )

        image = cv2.imread(
            str(image_path)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

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

        image = torch.from_numpy(
            image
        ).permute(2, 0, 1).float() / 255.0

        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

        image = (image - mean) / std

        mask = torch.from_numpy(
            mask.astype(np.int64)
        )

        return image, mask


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, loader):

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_pixels = 0

    intersection = np.zeros(NUM_CLASSES)
    union = np.zeros(NUM_CLASSES)

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

        total_loss += loss.item()

        total_correct += (
            predictions == masks
        ).sum().item()

        total_pixels += masks.numel()

        # --------------------------------------------
        # Per-class intersection / union
        # --------------------------------------------

        pred_np = predictions.cpu().numpy()
        mask_np = masks.cpu().numpy()

        for class_id in range(NUM_CLASSES):

            pred_class = pred_np == class_id
            mask_class = mask_np == class_id

            intersection[class_id] += np.logical_and(
                pred_class,
                mask_class
            ).sum()

            union[class_id] += np.logical_or(
                pred_class,
                mask_class
            ).sum()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    pixel_accuracy = (
        total_correct / total_pixels
    )

    class_ious = []

    for class_id in range(NUM_CLASSES):

        if union[class_id] == 0:
            iou = float("nan")
        else:
            iou = (
                intersection[class_id]
                / union[class_id]
            )

        class_ious.append(iou)

    mean_iou = np.nanmean(
        class_ious
    )

    return (
        total_loss / len(loader),
        pixel_accuracy,
        mean_iou,
        class_ious
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("FaceAware AI - Test Evaluation")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = CelebAMaskDataset(
        DATASET_ROOT,
        "test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(
        f"Test images: {len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = BiSeNet(
        num_classes=NUM_CLASSES,
        backbone_name="resnet18"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(DEVICE)

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    loss, accuracy, miou, class_ious = evaluate(
        model,
        test_loader
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(
        f"Test Loss       : {loss:.4f}"
    )

    print(
        f"Pixel Accuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Mean IoU        : {miou:.4f}"
    )

    print()
    print("Per-class IoU")
    print("-" * 40)

    for class_id, iou in enumerate(class_ious):

        if np.isnan(iou):
            print(
                f"{class_id:2d} "
                f"{CLASS_NAMES[class_id]:15s} "
                "N/A"
            )
        else:
            print(
                f"{class_id:2d} "
                f"{CLASS_NAMES[class_id]:15s} "
                f"{iou:.4f}"
            )

    print("=" * 60)


if __name__ == "__main__":
    main()