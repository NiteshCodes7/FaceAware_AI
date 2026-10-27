import cv2
import numpy as np


class GlassesEnhancer:
    """
    Glare removal for glasses.

    Stage A  specular glare (blown-out white)  -> the real pixels are gone, so we
             REPLACE them: first from the mirrored, colour-matched opposite lens
             (the other eye), else with a texture-aware inpaint (FSR / Telea).
    Stage B  violet / lavender veil            -> physically subtract the veil
             (pixels still carry information, so we correct instead of replace).
    """

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

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
    def _violet_score(L, A, B, min_l=20.0):
        """Violet = low B (blue) AND high A (magenta); or very strong blue.
        Irises are bluish but have A near 0, so they score ~0."""
        sb = np.clip((8.0 - B) / 10.0, 0.0, 1.0)
        sa = np.clip((A - 8.0) / 10.0, 0.0, 1.0)
        deep_blue = np.clip((-B - 22.0) / 10.0, 0.0, 1.0)
        sl = np.clip((L - min_l) / 15.0, 0.0, 1.0)
        return np.maximum(sb * sa, deep_blue) * sl

    @staticmethod
    def _centroid(m):
        M = cv2.moments(m, binaryImage=True)
        return M["m10"] / M["m00"], M["m01"] / M["m00"]

    # ------------------------------------------------------------------
    # stage A
    # ------------------------------------------------------------------

    def _find_glare_core(self, image, lens_roi, white_l, min_area_frac, max_chroma=14.0):
        """Blown-out, near-colourless blobs inside the lens area.
        Coloured (violet/blue) glare is left for stage B, which corrects it
        instead of replacing it."""
        h, w = image.shape[:2]
        lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        chroma = np.hypot(A, B)
        core = ((L > white_l) & (chroma < max_chroma) & (lens_roi > 0)).astype(np.uint8)
        core = cv2.morphologyEx(core, cv2.MORPH_OPEN, self._ellipse(max(3, int(max(h, w) * 0.008))))

        lens_area = max(1, int(cv2.countNonZero(lens_roi)))
        min_area = max(12, int(min_area_frac * lens_area))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
        keep = np.zeros((h, w), np.uint8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == i] = 255
        return keep

    def _mirror_donor(self, image, lens, glare_core):
        """
        Mirror the *other* lens onto the glared lens.
        `lens` may contain 2 blobs (left + right lens).
        Returns (donor_image, glared_lens_mask) or (None, None).
        """
        n, labels, stats, _ = cv2.connectedComponentsWithStats(lens, connectivity=8)
        comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 50]
        if len(comps) < 2:
            return None, None

        # glared lens = component that holds most glare pixels
        hits = [int(np.count_nonzero((labels == i) & (glare_core > 0))) for i in comps]
        gi = comps[int(np.argmax(hits))]
        # donor lens = the largest remaining component with no glare
        others = [i for i in comps if i != gi]
        di = max(others, key=lambda i: stats[i, cv2.CC_STAT_AREA])

        g_mask = (labels == gi).astype(np.uint8) * 255
        d_mask = (labels == di).astype(np.uint8) * 255
        gx, gy = self._centroid(g_mask)
        dx, dy = self._centroid(d_mask)

        # horizontal flip about the midpoint between the lens centres
        M = np.float32([[-1, 0, gx + dx], [0, 1, gy - dy]])
        h, w = image.shape[:2]
        donor = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REFLECT)
        return donor, g_mask

    def _match_colour(self, donor, image, mask, ref_ring):
        """Match donor LAB mean/std to the target using a ring of clean pixels."""
        dl = cv2.cvtColor(donor.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
        il = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
        for c in range(3):
            s, d = il[..., c][ref_ring], dl[..., c][ref_ring]
            dl[..., c] = (dl[..., c] - d.mean()) * (s.std() / max(d.std(), 1e-3)) + s.mean()
        out = cv2.cvtColor(dl, cv2.COLOR_LAB2BGR)
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)

    def _remove_white_glare(self, image, lens, lens_roi, white_l, min_area_frac,
                            halo_frac, use_donor, max_chroma=14.0, donor_max_err=25.0):
        h, w = image.shape[:2]
        size = max(h, w)

        core = self._find_glare_core(image, lens_roi, white_l, min_area_frac, max_chroma)
        if not core.any():
            return image

        # Keep the mask TIGHT: core + a modest halo (the bluish bloom around it).
        # The old 2*grow+1 dilation was what ate the whole eye.
        halo = max(5, int(size * halo_frac))
        mask = cv2.dilate(core, self._ellipse(halo))
        mask = cv2.bitwise_and(mask, lens_roi)

        lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
        L = lab[..., 0]

        # ---- option 1: mirrored donor eye ------------------------------
        if use_donor:
            donor, g_mask = self._mirror_donor(image, lens, core)
            if donor is None:
                print("[donor skipped] lens mask needs 2 separate lens blobs")
            if donor is not None:
                ring = (cv2.dilate(mask, self._ellipse(halo)) > 0) & (mask == 0) \
                       & (g_mask > 0) & (L < white_l)
                if ring.sum() > 30:
                    donor = self._match_colour(donor, image, mask, ring)
                    # sanity check: if the mirrored lens doesn't resemble this one around
                    # the glare (tilted / turned head, different lighting) don't use it
                    dg = cv2.cvtColor(donor, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    ig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    err = float(np.abs(dg - ig)[ring].mean())
                    if err > donor_max_err:
                        print(f"[donor skipped] mirrored lens differs too much (err {err:.1f} > {donor_max_err})")
                        donor = None
                if donor is not None and ring.sum() > 30:
                    ys, xs = np.where(mask > 0)
                    centre = (int((xs.min() + xs.max()) // 2), int((ys.min() + ys.max()) // 2))
                    # seamlessClone needs the mask bbox to stay inside the image
                    pad = 2
                    if xs.min() > pad and ys.min() > pad and xs.max() < w - pad and ys.max() < h - pad:
                        return cv2.seamlessClone(donor, image, mask, centre, cv2.NORMAL_CLONE)
                    print("[donor skipped] glare mask touches the image border")
                elif donor is not None:
                    print("[donor skipped] too few clean pixels around glare (lower white_l / raise white_halo)")

        # ---- option 2: texture-aware inpaint (no eye to copy from) -----
        m = cv2.dilate(core, self._ellipse(5))
        m = cv2.bitwise_and(m, lens_roi)
        if hasattr(getattr(cv2, "xphoto", None), "inpaint"):
            try:
                out = np.zeros_like(image)
                cv2.xphoto.inpaint(image, 255 - m, out, cv2.xphoto.INPAINT_FSR_FAST)
                return out
            except Exception:
                pass
        return cv2.inpaint(image, m, max(3, int(size * 0.008)), cv2.INPAINT_TELEA)

    # ------------------------------------------------------------------
    # main
    # ------------------------------------------------------------------

    def _remove_glare_classical(
        self,
        image,
        glare_mask,
        lens_mask=None,
        strength=0.5,
        chroma_start=2.0,
        chroma_full=12.0,
        luma_gain=0.85,
        shadow_lift=0.0,       # 0..1: brighten the corrected glare area so detail behind the lens shows (0 = off)
        auto_detect=True,
        violet_thresh=0.12,
        remove_white=True,
        white_l=80.0,          # L above this inside the lens = blown glare core
        white_min_area=0.003,  # min blob size as fraction of lens area
        white_halo=0.09,       # halo around glare core, fraction of image size (0.09 tested; >0.12 copies extra catchlights)
        use_donor=True,        # copy the other eye/lens if a 2-lens mask is given
        white_strength=1.0,    # stage A is a replacement, so default is full strength
        grow_frac=0.08,        # how far (fraction of image size) violet/blue glare may extend past the lens mask
        violet_min_l=20.0,     # ignore pixels darker than this. For strong blue-screen glare try 5 (but it can recolour dark blue irises)
        white_chroma=14.0,     # max colourfulness of "white" glare; coloured glare goes to stage B
    ):
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength == 0:
            return image.copy()

        h, w = image.shape[:2]
        size = max(h, w)

        glare = self._binary(glare_mask, (h, w))
        lens = self._binary(lens_mask, (h, w)) if lens_mask is not None else None

        lens_roi = None
        if lens is not None:
            lens_roi = cv2.dilate(lens, self._ellipse(max(5, int(size * 0.04))))

        # ---- Stage A ----------------------------------------------------
        if remove_white and lens_roi is not None:
            fixed = self._remove_white_glare(
                image, lens, lens_roi, white_l, white_min_area, white_halo, use_donor, white_chroma
            )
            ws = float(np.clip(white_strength, 0.0, 1.0))
            image = np.clip(
                image.astype(np.float32) * (1 - ws) + fixed.astype(np.float32) * ws, 0, 255
            ).astype(np.uint8)

        # ---- Stage B (unchanged) ---------------------------------------
        img_f = image.astype(np.float32) / 255.0
        lab = cv2.cvtColor(img_f, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        kernel = self._ellipse(max(3, int(size * 0.003)))

        user_glare = glare.copy()          # what the caller explicitly marked
        violet = self._violet_score(L, A, B, violet_min_l)
        violet = cv2.GaussianBlur(violet, (0, 0), max(1.0, size * 0.0008))

        region = lens_roi          # where stage B is allowed to work
        if auto_detect:
            auto = (violet > violet_thresh).astype(np.uint8) * 255
            if lens_roi is not None:
                # Lens masks (circles) rarely match real lens shapes. Instead of clipping
                # the glare to the mask, keep every violet/blue patch that touches the lens
                # area, allowing it to extend a limited distance past the mask.
                auto = cv2.morphologyEx(auto, cv2.MORPH_OPEN, self._ellipse(3))
                limit = cv2.dilate(lens_roi, self._ellipse(max(5, int(size * grow_frac))))
                auto = cv2.bitwise_and(auto, limit)
                n, labels, stats, _ = cv2.connectedComponentsWithStats(auto, connectivity=8)
                grown = np.zeros_like(auto)
                for i in range(1, n):
                    comp = labels == i
                    if np.count_nonzero(comp & (lens_roi > 0)) >= 20:
                        grown[comp] = 255
                auto = grown
                region = cv2.bitwise_or(lens_roi, cv2.dilate(auto, self._ellipse(max(5, int(size * 0.01)))))
            else:
                near_user = cv2.dilate(glare, self._ellipse(max(5, int(size * 0.05))))
                auto = cv2.bitwise_and(auto, near_user)
            auto = cv2.morphologyEx(auto, cv2.MORPH_OPEN, self._ellipse(3))
            auto = cv2.morphologyEx(auto, cv2.MORPH_CLOSE, self._ellipse(max(5, int(size * 0.01))))
            glare = cv2.bitwise_or(glare, auto)

        if region is not None:
            glare = cv2.bitwise_and(glare, region)
        if not glare.any():
            return image.copy()

        mask = cv2.dilate(glare, kernel, iterations=4)
        if region is not None:
            mask = cv2.bitwise_and(mask, region)

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
        ref_map = self._smooth_fill(lab, ref.astype(np.float32), sigma=size * 0.03, fallback=med)
        ref_L, ref_A, ref_B = cv2.split(ref_map)

        d = np.hypot(A - ref_A, B - ref_B)
        t = np.clip((d - chroma_start) / (chroma_full - chroma_start), 0.0, 1.0)
        wc = t * t * (3.0 - 2.0 * t)
        # "differs from skin" is only trusted inside the caller's own glare mask;
        # elsewhere it would also pull irises, lashes and eye whites toward skin colour.
        if user_glare.any():
            near_user = cv2.dilate(user_glare, self._ellipse(max(5, int(size * 0.02))))
            wc = wc * (near_user > 0)
        else:
            wc = wc * 0.0
        wc = np.maximum(wc, np.clip(violet * 2.0, 0.0, 1.0))
        wc = cv2.GaussianBlur(wc, (0, 0), max(1.0, size * 0.0008))
        dark_keep = np.clip((L - 12.0) / 18.0, 0.0, 1.0)
        wc = wc * (0.35 + 0.65 * dark_keep)

        Ls = cv2.GaussianBlur(L, (0, 0), max(1.5, size * 0.002))
        excess = np.clip(Ls - ref_L - 2.0, 0.0, None)
        L2 = L - excess * wc * luma_gain

        bright = np.clip((L2 - ref_L - 15.0) / 50.0, 0.0, 0.8)
        A2 = A + (ref_A * (1.0 - bright) - A) * wc
        B2 = B + (ref_B * (1.0 - bright) - B) * wc

        corrected = cv2.cvtColor(cv2.merge([L2, A2, B2]), cv2.COLOR_LAB2BGR)
        corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

        blown = ((L2 > 95) & (mask > 0)).astype(np.uint8) * 255
        if blown.any():
            blown = cv2.dilate(blown, np.ones((3, 3), np.uint8), iterations=1)
            corrected = cv2.inpaint(corrected, blown, 5, cv2.INPAINT_TELEA)

        alpha = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), max(1.5, size * 0.002))
        alpha = (np.clip(alpha, 0.0, 1.0) * strength)[..., None]
        result = image.astype(np.float32) * (1.0 - alpha) + corrected.astype(np.float32) * alpha
        result = np.clip(result, 0, 255).astype(np.uint8)

        if shadow_lift > 0:
            # Lift shadows only where glare was removed. Brightness is raised with a gamma curve
            # on L (colour untouched), so dark detail behind the lens (eyes, lashes) shows up.
            lift = float(np.clip(shadow_lift, 0.0, 1.0))
            lr = cv2.cvtColor(result.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
            Lr = lr[..., 0]
            lw = np.clip(wc * 1.5, 0.0, 1.0)
            lw = cv2.GaussianBlur(lw, (0, 0), max(1.5, size * 0.003)) * (mask > 0)
            Ln = 100.0 * np.power(np.clip(Lr, 0.0, 100.0) / 100.0, 1.0 - 0.55 * lift)
            lr[..., 0] = Lr * (1.0 - lw * lift) + Ln * (lw * lift)
            result = np.clip(cv2.cvtColor(lr, cv2.COLOR_LAB2BGR) * 255.0, 0, 255).astype(np.uint8)
        return result

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------

    _lama_model = None

    def remove_glare(self, image, glare_mask, lens_mask=None, generative=None,
                     lama_grow=0.01, **kwargs):
        """
        Default (generative=None): classical pipeline only. Every output pixel comes
        from the input photo; nothing is invented.

        generative="lama": the classical pipeline first finds WHERE glare is, then the
        LaMa neural inpainter repaints those pixels. The repainted pixels are generated
        by AI, not recovered from the photo. After the call, `self.generative_used`
        is True when LaMa actually changed the image.

        All other keyword arguments (strength, auto_detect, violet_min_l, ...) go to the
        classical pipeline.
        """
        self.generative_used = False
        if generative is None:
            return self._remove_glare_classical(image, glare_mask, lens_mask, **kwargs)
        if generative != "lama":
            raise ValueError('generative must be None or "lama"')

        # 1. find the glare area with the classical pipeline at full strength
        probe_kwargs = dict(kwargs)
        probe_kwargs["strength"] = 1.0
        probe_kwargs["shadow_lift"] = 0.0
        classical = self._remove_glare_classical(image, glare_mask, lens_mask, **probe_kwargs)
        diff = np.abs(classical.astype(np.int16) - image.astype(np.int16)).max(axis=2)
        mask = (diff > 10).astype(np.uint8) * 255
        h, w = image.shape[:2]
        size = max(h, w)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._ellipse(3))
        mask = cv2.dilate(mask, self._ellipse(max(3, int(size * lama_grow))))
        if not mask.any():
            return classical            # no glare found, nothing to repaint

        # 2. LaMa on a crop around the glare (upscaled so the model has enough pixels)
        out = self._run_lama(image, mask)
        self.generative_used = True
        print("[generative] LaMa repainted the glare area - those pixels are AI-generated.")
        return out

    def _run_lama(self, image, mask):
        try:
            from PIL import Image
            from simple_lama_inpainting import SimpleLama
        except ImportError as e:
            raise ImportError(
                "generative='lama' needs: pip install simple-lama-inpainting"
            ) from e
        if GlassesEnhancer._lama_model is None:
            GlassesEnhancer._lama_model = SimpleLama()   # downloads weights on first use
        lama = GlassesEnhancer._lama_model

        h, w = image.shape[:2]
        ys, xs = np.where(mask > 0)
        m = max(24, int(0.35 * max(ys.max() - ys.min(), xs.max() - xs.min())))
        y0, y1 = max(0, ys.min() - m), min(h, ys.max() + m + 1)
        x0, x1 = max(0, xs.min() - m), min(w, xs.max() + m + 1)
        crop, cmask = image[y0:y1, x0:x1], mask[y0:y1, x0:x1]

        scale = int(np.clip(np.ceil(512 / max(crop.shape[:2])), 1, 4))
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        bigm = cv2.resize(cmask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        res = lama(Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB)), Image.fromarray(bigm))
        res = cv2.cvtColor(np.array(res), cv2.COLOR_RGB2BGR)[: big.shape[0], : big.shape[1]]
        res = cv2.resize(res, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA)

        # feathered paste: only the masked area changes
        a = cv2.GaussianBlur((cmask > 0).astype(np.float32), (0, 0), max(1.0, max(h, w) * 0.003))[..., None]
        out = image.copy()
        out[y0:y1, x0:x1] = np.clip(crop * (1 - a) + res * a, 0, 255).astype(np.uint8)
        return out