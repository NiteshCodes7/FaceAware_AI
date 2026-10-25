import warnings
import cv2
import numpy as np

GAIN = 1.0          # overall effect multiplier: 0.7 = subtler, 1.5 = stronger
BLACK_FLOOR = 6     # darkest L value hairs are pushed toward (0 = pure black)

BROW_CLASSES = (6, 7)


class EyebrowEnhancer:
    """
    Makes eyebrows denser, darker (blackish) and more defined.

    Pass either:
      - mask:      BiSeNet class map, or a binary 0/255 mask (auto-detected)
      - brow_mask: a ready binary mask (0/255)
    """

    def __init__(self, gain=GAIN, brow_classes=BROW_CLASSES, black_floor=BLACK_FLOOR):
        self.brow_classes = tuple(brow_classes)
        self.gain = gain
        self.black_floor = black_floor

    @staticmethod
    def _ellipse(k):
        k = int(k) | 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    @staticmethod
    def _is_binary_255(m):
        u = np.unique(m)
        return u.size <= 2 and set(u.tolist()) <= {0, 255}

    def enhance(self, image, mask=None, strength=0.5, brow_mask=None, debug=False):
        if image is None or image.ndim != 3:
            raise ValueError("image must be a valid BGR image")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be between 0.0 and 1.0")
        if strength == 0.0:
            return image.copy()

        h, w = image.shape[:2]
        size = max(h, w)

        # ---- mask handling -------------------------------------------------
        if brow_mask is None and mask is not None and mask.ndim == 2 \
                and self._is_binary_255(mask):
            brow_mask, mask = mask, None

        if brow_mask is not None:
            if brow_mask.ndim == 3:
                brow_mask = brow_mask[..., 0]
            if brow_mask.shape != (h, w):
                brow_mask = cv2.resize(brow_mask.astype(np.uint8), (w, h),
                                       interpolation=cv2.INTER_NEAREST)
            brow_mask = np.where(brow_mask > 127, 255, 0).astype(np.uint8)
        else:
            if mask is None or mask.ndim != 2:
                raise ValueError("mask must be a 2D segmentation mask")
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h),
                                  interpolation=cv2.INTER_NEAREST)
            brow_mask = np.isin(mask, self.brow_classes).astype(np.uint8) * 255
            if not brow_mask.any():
                warnings.warn(
                    f"No pixels for brow classes {self.brow_classes}. "
                    f"Classes present: {np.unique(mask).tolist()}",
                    stacklevel=2,
                )

        coverage = np.count_nonzero(brow_mask) / (h * w)
        if debug:
            print(f"[brow] mask coverage: {coverage:.4%}")
        if coverage > 0.05:
            warnings.warn(
                f"Brow mask covers {coverage:.1%} of the image - likely the "
                f"wrong mask. Brows are usually ~0.3-2%.", stacklevel=2)
        if not np.any(brow_mask):
            return image.copy()

        # Cover stray hairs just outside the segmentation
        brow_mask = cv2.dilate(brow_mask, self._ellipse(max(3, int(size * 0.004))))
        inside = brow_mask > 0

        # ---- LAB working space --------------------------------------------
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32)
        A = lab[:, :, 1].astype(np.float32)
        B = lab[:, :, 2].astype(np.float32)

        d = float(np.clip(strength * self.gain, 0.0, 1.0)) ** 0.7  # visible even at 0.25

        # ---- hair weight: how much darker than surrounding skin ------------
        outer = cv2.dilate(brow_mask, self._ellipse(max(7, int(size * 0.03))))
        ring = (outer > 0) & ~inside
        if ring.sum() > 50:
            skin_L = float(np.median(L[ring]))
        else:
            skin_L = float(np.percentile(L[inside], 90))

        hair_w = np.clip((skin_L - L) / (skin_L * 0.45 + 1e-6), 0.0, 1.0) ** 0.8
        weight = 0.3 + 0.7 * hair_w          # gaps get some darkening, hairs get full

        # 1. Density: remove pale gaps between hairs
        k = max(5, int(size * 0.0035))
        opened = cv2.morphologyEx(L, cv2.MORPH_OPEN, self._ellipse(k))
        opened = cv2.GaussianBlur(opened, (0, 0), max(1.0, k / 4))
        fill = float(np.clip(0.8 * d, 0.0, 0.95))
        L = L + (opened - L) * fill

        # 2. Blackening: push hair pixels toward near-black
        floor = float(self.black_floor)
        L = L - (L - floor) * np.clip(0.92 * d * weight, 0.0, 0.97)

        # 3. Remove brown tint: pull chroma toward neutral
        chroma_cut = np.clip(0.75 * d * weight, 0.0, 0.9)
        A = 128.0 + (A - 128.0) * (1.0 - chroma_cut)
        B = 128.0 + (B - 128.0) * (1.0 - chroma_cut)

        # 4. Definition: fine-detail boost
        L = np.clip(L, 0, 255)
        blurred = cv2.GaussianBlur(L, (0, 0), 1.0)
        L = L + (L - blurred) * (0.5 + 0.8 * strength)

        L = np.clip(L, 0, 255).astype(np.uint8)
        A = np.clip(A, 0, 255).astype(np.uint8)
        B = np.clip(B, 0, 255).astype(np.uint8)
        enhanced = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

        # 5. Soft blend inside the mask only
        alpha = cv2.GaussianBlur(brow_mask.astype(np.float32) / 255.0, (0, 0), max(1.0, size * 0.0012))
        alpha = (np.clip(alpha, 0, 1) * (0.85 + 0.15 * strength))[:, :, None]

        result = image.astype(np.float32) * (1.0 - alpha) + enhanced.astype(np.float32) * alpha
        return np.clip(result, 0, 255).astype(np.uint8)