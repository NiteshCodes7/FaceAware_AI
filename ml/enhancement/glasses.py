import cv2
import numpy as np


class GlassesEnhancer:

    @staticmethod
    def _binary(mask, size):
        h, w = size
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return np.where(mask > 127, 255, 0).astype(np.uint8)

    @staticmethod
    def _ellipse(k):
        k = int(k) | 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    @staticmethod
    def _smooth_fill(lab, weight, sigma, fallback, down=4):
        h, w = lab.shape[:2]
        sh, sw = max(1, h // down), max(1, w // down)

        vals = cv2.resize(lab * weight[..., None], (sw, sh), interpolation=cv2.INTER_AREA)
        wts = cv2.resize(weight, (sw, sh), interpolation=cv2.INTER_AREA)

        s = max(1.0, sigma / down)
        vals = cv2.GaussianBlur(vals, (0, 0), s)
        wts = cv2.GaussianBlur(wts, (0, 0), s)

        fill = vals / np.maximum(wts[..., None], 1e-6)
        conf = np.clip(wts / 0.05, 0.0, 1.0)[..., None]
        fill = fill * conf + fallback.reshape(1, 1, 3) * (1.0 - conf)

        return cv2.resize(fill, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _violet_score(L, A, B):
        """Soft 0..1 score for violet / pale lavender reflection."""
        sb = np.clip((8.0 - B) / 10.0, 0.0, 1.0)
        sa = np.clip((A + 2.0) / 6.0, 0.0, 1.0)
        sl = np.clip((L - 20.0) / 15.0, 0.0, 1.0)
        return sb * sa * sl

    def remove_glare(
        self,
        image,
        glare_mask,
        lens_mask=None,
        strength=0.5,
        chroma_start=2.0,
        chroma_full=12.0,
        luma_gain=0.85,
        auto_detect=True,
        violet_thresh=0.12,
    ):
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength == 0:
            return image.copy()

        h, w = image.shape[:2]
        size = max(h, w)

        # 1. Masks ----------------------------------------------------
        glare = self._binary(glare_mask, (h, w))
        lens = self._binary(lens_mask, (h, w)) if lens_mask is not None else None

        # lens used only as a loose ROI so imperfect lens masks still work
        lens_roi = None
        if lens is not None:
            lens_roi = cv2.dilate(lens, self._ellipse(max(5, int(size * 0.04))))

        # 2. LAB (float: L 0-100, a/b neutral = 0) --------------------
        img_f = image.astype(np.float32) / 255.0
        lab = cv2.cvtColor(img_f, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        kernel = self._ellipse(max(3, int(size * 0.003)))

        # 3. Auto-detect violet glare missed by the supplied mask -----
        violet = self._violet_score(L, A, B)
        violet = cv2.GaussianBlur(violet, (0, 0), max(1.0, size * 0.0008))

        if auto_detect:
            auto = (violet > violet_thresh).astype(np.uint8) * 255
            if lens_roi is not None:
                auto = cv2.bitwise_and(auto, lens_roi)
            else:
                near_user = cv2.dilate(glare, self._ellipse(max(5, int(size * 0.05))))
                auto = cv2.bitwise_and(auto, near_user)

            auto = cv2.morphologyEx(auto, cv2.MORPH_OPEN, self._ellipse(3))
            auto = cv2.morphologyEx(auto, cv2.MORPH_CLOSE, self._ellipse(max(5, int(size * 0.01))))
            glare = cv2.bitwise_or(glare, auto)

        if lens_roi is not None:
            glare = cv2.bitwise_and(glare, lens_roi)
        if not glare.any():
            return image.copy()

        mask = cv2.dilate(glare, kernel, iterations=4)
        if lens_roi is not None:
            mask = cv2.bitwise_and(mask, lens_roi)

        # 4. Reference ring (clean skin around glare) -----------------
        near = cv2.dilate(mask, kernel, iterations=2)
        far = cv2.dilate(mask, self._ellipse(max(5, int(size * 0.03))))
        ring = (far > 0) & (near == 0) & (violet < 0.1)

        if lens_roi is not None:
            in_lens = ring & (lens_roi > 0)
            if in_lens.sum() > 500:
                ring = in_lens
        if ring.sum() < 50:
            ring = (mask == 0) & (violet < 0.1)

        med = np.median(lab[ring], axis=0)
        dist = np.hypot(A - med[1], B - med[2])
        keep = ring & (dist < 15) & (np.abs(L - med[0]) < 25)
        ref = keep if keep.sum() >= 50 else ring

        med = np.median(lab[ref], axis=0).astype(np.float32)
        ref_map = self._smooth_fill(
            lab, ref.astype(np.float32), sigma=size * 0.03, fallback=med
        )
        ref_L, ref_A, ref_B = cv2.split(ref_map)

        # 5. Per-pixel glare weight -----------------------------------
        d = np.hypot(A - ref_A, B - ref_B)
        t = np.clip((d - chroma_start) / (chroma_full - chroma_start), 0.0, 1.0)
        wc = t * t * (3.0 - 2.0 * t)
        wc = np.maximum(wc, np.clip(violet * 2.0, 0.0, 1.0))
        wc = cv2.GaussianBlur(wc, (0, 0), max(1.0, size * 0.0008))

        dark_keep = np.clip((L - 12.0) / 18.0, 0.0, 1.0)
        wc = wc * (0.35 + 0.65 * dark_keep)

        # 6. Luminance: remove only excess brightness -----------------
        Ls = cv2.GaussianBlur(L, (0, 0), max(1.5, size * 0.002))
        excess = np.clip(Ls - ref_L - 2.0, 0.0, None)
        L2 = L - excess * wc * luma_gain

        # 7. Chroma: pull toward clean skin chroma --------------------
        bright = np.clip((L2 - ref_L - 15.0) / 50.0, 0.0, 0.8)
        A2 = A + (ref_A * (1.0 - bright) - A) * wc
        B2 = B + (ref_B * (1.0 - bright) - B) * wc

        corrected = cv2.cvtColor(cv2.merge([L2, A2, B2]), cv2.COLOR_LAB2BGR)
        corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

        # 8. Rebuild blown highlights ---------------------------------
        blown = ((L2 > 95) & (mask > 0)).astype(np.uint8) * 255
        if blown.any():
            blown = cv2.dilate(blown, np.ones((3, 3), np.uint8), iterations=1)
            corrected = cv2.inpaint(corrected, blown, 5, cv2.INPAINT_TELEA)

        # 9. Soft blend -----------------------------------------------
        alpha = cv2.GaussianBlur(
            (mask > 0).astype(np.float32), (0, 0), max(1.5, size * 0.002)
        )
        alpha = (np.clip(alpha, 0.0, 1.0) * strength)[..., None]

        result = image.astype(np.float32) * (1.0 - alpha) + corrected.astype(np.float32) * alpha
        return np.clip(result, 0, 255).astype(np.uint8)