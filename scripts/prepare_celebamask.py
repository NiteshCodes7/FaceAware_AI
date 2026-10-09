import os
import cv2
import numpy as np


LABELS = [
    "skin",
    "nose",
    "eye_g",
    "l_eye",
    "r_eye",
    "l_brow",
    "r_brow",
    "l_ear",
    "r_ear",
    "mouth",
    "u_lip",
    "l_lip",
    "hair",
    "hat",
    "ear_r",
    "neck_l",
    "neck",
    "cloth"
]


RAW_MASK_DIR = "datasets/celebAMask/raw/CelebAMask-HQ-mask-anno"

OUTPUT_DIR = "datasets/celebAMask/masks"


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


NUM_IMAGES = 30000


for image_id in range(NUM_IMAGES):

    folder_id = image_id // 2000

    folder = os.path.join(
        RAW_MASK_DIR,
        str(folder_id)
    )

    final_mask = np.zeros(
        (512, 512),
        dtype=np.uint8
    )

    for label_index, label in enumerate(LABELS):

        filename = (
            f"{image_id:05d}_{label}.png"
        )

        path = os.path.join(
            folder,
            filename
        )

        if not os.path.exists(path):
            continue

        mask = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            continue

        final_mask[mask > 0] = label_index + 1

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{image_id}.png"
    )

    cv2.imwrite(
        output_path,
        final_mask
    )

    if image_id % 500 == 0:

        print(
            f"Processed {image_id}/{NUM_IMAGES}"
        )


print("Mask preparation complete.")