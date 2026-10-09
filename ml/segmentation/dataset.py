import os

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


class CelebAMaskDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        image_ids,
        image_size=256
    ):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ids = image_ids
        self.image_size = image_size


    def __len__(self):

        return len(self.image_ids)


    def __getitem__(self, index):

        image_id = self.image_ids[index]

        image_path = os.path.join(
            self.image_dir,
            f"{image_id}.jpg"
        )

        mask_path = os.path.join(
            self.mask_dir,
            f"{image_id}.png"
        )


        # -------------------------
        # Read image
        # -------------------------

        image = cv2.imread(
            image_path
        )

        if image is None:
            raise RuntimeError(
                f"Could not read {image_path}"
            )


        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # -------------------------
        # Read mask
        # -------------------------

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise RuntimeError(
                f"Could not read {mask_path}"
            )


        # -------------------------
        # Resize
        # -------------------------

        image = cv2.resize(
            image,
            (
                self.image_size,
                self.image_size
            ),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (
                self.image_size,
                self.image_size
            ),
            interpolation=cv2.INTER_NEAREST
        )


        # -------------------------
        # Normalize image
        # -------------------------

        image = image.astype(
            np.float32
        ) / 255.0


        # H,W,C → C,H,W

        image = np.transpose(
            image,
            (2, 0, 1)
        )


        # -------------------------
        # Convert to tensors
        # -------------------------

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.long
        )


        return image, mask