import cv2
import numpy as np
import torch
from pathlib import Path

from models.bisenet import BiSeNet


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 512
NUM_CLASSES = 19

IMAGE_PATH = Path(
    "../datasets/celebAMask/split/test/images/00103.jpg"
)

MASK_PATH = Path(
    "../datasets/celebAMask/split/test/masks/00103.png"
)

MODEL_PATH = Path(
    "../models/face_parser/training/"
    "bisenet_celebamask_best.pt"
)

OUTPUT_PATH = Path(
    "../datasets/celebAMask/prediction_visualization.jpg"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Colors
# ============================================================

COLORS = {
    0: (0, 0, 0),
    1: (255, 180, 180),
    2: (0, 255, 255),
    3: (255, 0, 0),
    4: (0, 255, 0),
    5: (0, 200, 0),
    6: (255, 255, 0),
    7: (200, 200, 0),
    8: (255, 0, 255),
    9: (200, 0, 255),
    10: (0, 128, 255),
    11: (0, 0, 255),
    12: (100, 0, 255),
    13: (128, 64, 0),
    14: (0, 128, 128),
    15: (128, 0, 128),
    16: (64, 128, 128),
    17: (64, 64, 255),
    18: (128, 128, 128),
}


# ============================================================
# Colorize mask
# ============================================================

def colorize(mask):

    colored = np.zeros(
        (mask.shape[0], mask.shape[1], 3),
        dtype=np.uint8
    )

    for class_id, color in COLORS.items():
        colored[mask == class_id] = color

    return colored


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("FaceAware AI - Prediction Visualization")
    print("=" * 60)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = BiSeNet(
        num_classes=NUM_CLASSES,
        backbone_name="resnet18"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(
        str(IMAGE_PATH)
    )

    if image is None:
        raise RuntimeError(
            f"Could not load {IMAGE_PATH}"
        )

    original = image.copy()

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    resized = cv2.resize(
        rgb,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    tensor = (
        torch.from_numpy(resized)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    tensor = (
        (tensor - mean) / std
    )

    tensor = tensor.unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(tensor)[0]

        prediction = torch.argmax(
            output,
            dim=1
        )[0].cpu().numpy().astype(np.uint8)

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    ground_truth = cv2.imread(
        str(MASK_PATH),
        cv2.IMREAD_GRAYSCALE
    )

    ground_truth = cv2.resize(
        ground_truth,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    # --------------------------------------------------------
    # Color masks
    # --------------------------------------------------------

    predicted_color = colorize(
        prediction
    )

    ground_truth_color = colorize(
        ground_truth
    )

    # --------------------------------------------------------
    # Resize original
    # --------------------------------------------------------

    original = cv2.resize(
        original,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    # --------------------------------------------------------
    # Overlays
    # --------------------------------------------------------

    prediction_overlay = cv2.addWeighted(
        original,
        0.55,
        predicted_color,
        0.45,
        0
    )

    ground_truth_overlay = cv2.addWeighted(
        original,
        0.55,
        ground_truth_color,
        0.45,
        0
    )

    # --------------------------------------------------------
    # Create 2x2 comparison
    # --------------------------------------------------------

    top = np.hstack([
        original,
        ground_truth_overlay
    ])

    bottom = np.hstack([
        prediction_overlay,
        predicted_color
    ])

    comparison = np.vstack([
        top,
        bottom
    ])

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    cv2.putText(
        comparison,
        "Original",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        comparison,
        "Ground Truth",
        (IMAGE_SIZE + 20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        comparison,
        "Prediction Overlay",
        (20, IMAGE_SIZE + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        comparison,
        "Prediction Mask",
        (IMAGE_SIZE + 20, IMAGE_SIZE + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    cv2.imwrite(
        str(OUTPUT_PATH),
        comparison
    )

    print()
    print("Visualization saved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()