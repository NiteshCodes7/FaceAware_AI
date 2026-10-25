import argparse
from pathlib import Path

import cv2
import numpy as np

from segmentation.parser import FaceParser


LEFT_BROW_CLASS = 6
RIGHT_BROW_CLASS = 7


def main():
    parser = argparse.ArgumentParser(
        description="Visualize BiSeNet eyebrow segmentation masks."
    )
    parser.add_argument(
        "--image",
        default="test.jpeg",
        help="Input image path (default: test.jpeg)",
    )
    parser.add_argument(
        "--model",
        default="../models/face_parser/training/bisenet_celebamask_best.pt",
        help="FaceParser model path",
    )
    parser.add_argument(
        "--output",
        default="outputs/eyebrow_mask_test",
        help="Output directory",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {image_path}")

    print(f"Image: {image_path} | dimensions: {image.shape[1]}x{image.shape[0]}")
    print(f"Model: {model_path}")
    print("Loading FaceParser...")

    face_parser = FaceParser(str(model_path))
    prediction = face_parser.predict(image)

    if prediction.shape != image.shape[:2]:
        prediction = cv2.resize(
            prediction.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # Save the combined eyebrow mask and each class independently.
    left_mask = (prediction == LEFT_BROW_CLASS).astype(np.uint8) * 255
    right_mask = (prediction == RIGHT_BROW_CLASS).astype(np.uint8) * 255
    combined_mask = cv2.bitwise_or(left_mask, right_mask)

    # Use distinct colors for the two predicted classes.
    overlay = image.copy()

    overlay[left_mask > 0] = (
        0.35 * overlay[left_mask > 0]
        + 0.65 * np.array([0, 255, 0], dtype=np.float32)
    ).astype(np.uint8)

    overlay[right_mask > 0] = (
        0.35 * overlay[right_mask > 0]
        + 0.65 * np.array([255, 0, 255], dtype=np.float32)
    ).astype(np.uint8)

    # Also create a contour view to inspect mask boundaries.
    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_view = image.copy()
    cv2.drawContours(contour_view, contours, -1, (0, 255, 255), 2)

    # Report class distribution to help detect missing/incorrect class IDs.
    total_pixels = prediction.size

    left_pixels = int(np.count_nonzero(left_mask))

    right_pixels = int(np.count_nonzero(right_mask))

    print(f"Class {LEFT_BROW_CLASS} pixels: {left_pixels} "
          f"({100 * left_pixels / total_pixels:.5f}% of image)")
    print(f"Class {RIGHT_BROW_CLASS} pixels: {right_pixels} "
          f"({100 * right_pixels / total_pixels:.5f}% of image)")
    print(f"Combined eyebrow pixels: {np.count_nonzero(combined_mask)}")

    outputs = {
        "eyebrow_mask_left.png": left_mask,
        "eyebrow_mask_right.png": right_mask,
        "eyebrow_mask_combined.png": combined_mask,
        "eyebrow_overlay.png": overlay,
        "eyebrow_contours.png": contour_view,
    }

    for filename, output in outputs.items():
        destination = output_dir / filename
        if not cv2.imwrite(str(destination), output):
            raise OSError(f"Failed to save: {destination}")
        print(f"Saved: {destination}")

    # A convenient side-by-side preview, resized for viewing only.
    preview_width = 900
    scale = min(1.0, preview_width / image.shape[1])
    preview_size = (
        int(image.shape[1] * scale),
        int(image.shape[0] * scale),
    )

    original_preview = cv2.resize(image, preview_size)
    overlay_preview = cv2.resize(overlay, preview_size)
    mask_preview = cv2.cvtColor(
        cv2.resize(combined_mask, preview_size),
        cv2.COLOR_GRAY2BGR,
    )
    
    preview = np.hstack([
        original_preview,
        overlay_preview,
        mask_preview,
    ])
    preview_path = output_dir / "eyebrow_mask_preview.jpg"
    if not cv2.imwrite(str(preview_path), preview):
        raise OSError(f"Failed to save: {preview_path}")
    print(f"Saved: {preview_path}")
    print("Eyebrow mask test completed.")


if __name__ == "__main__":
    main()
