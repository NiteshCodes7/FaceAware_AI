import cv2
import numpy as np

# ============================================================
# Segmentation
# ============================================================

from segmentation.parser import FaceParser
from segmentation.region_extractor import RegionExtractor

# ============================================================
# Landmarks
# ============================================================

from landmarks.face_landmarks import FaceLandmarks
from segmentation.region_masks import RegionMasks

# ============================================================
# Enhancement
# ============================================================

from enhancement.skin import smooth_skin
from enhancement.eyes import EyeEnhancer
from enhancement.iris import IrisEnhancer
from enhancement.eye_bag import EyeBagEnhancer
from enhancement.eye_bag_mask import EyeBagMask
from enhancement.teeth import TeethEnhancer
from enhancement.glasses import GlassesEnhancer
from enhancement.lens_detector import LensDetector
from enhancement.wrinkles import WrinkleEnhancer


# ============================================================
# MediaPipe landmark indices
# ============================================================

LEFT_EYE = [
    33, 160, 158, 133, 153, 144
]

RIGHT_EYE = [
    362, 385, 387, 263, 373, 380
]

LEFT_IRIS = [
    474, 475, 476, 477
]

RIGHT_IRIS = [
    469, 470, 471, 472
]

INNER_MOUTH = [
    78, 95, 88, 178, 87, 14,
    317, 402, 318, 324, 308, 415,
    310, 311, 312, 13, 82, 81,
    80
]


# ============================================================
# Utility
# ============================================================

def combine_masks(*masks):

    result = np.zeros_like(masks[0])

    for mask in masks:
        result = cv2.bitwise_or(result, mask)

    return result


# ============================================================
# FaceAware Pipeline
# ============================================================

class FaceAwarePipeline:

    def __init__(self, model_path):

        print("Initializing FaceAware Pipeline...")

        # ----------------------------------------------------
        # Face parsing
        # ----------------------------------------------------

        self.face_parser = FaceParser(model_path)

        # ----------------------------------------------------
        # Facial landmarks
        # ----------------------------------------------------

        self.landmarks = FaceLandmarks()

        # ----------------------------------------------------
        # Enhancement modules
        # ----------------------------------------------------

        self.eye_enhancer = EyeEnhancer()
        self.iris_enhancer = IrisEnhancer()
        self.eye_bag_enhancer = EyeBagEnhancer()
        self.teeth_enhancer = TeethEnhancer()
        self.glasses_enhancer = GlassesEnhancer()
        self.wrinkle_enhancer = WrinkleEnhancer()

        # ----------------------------------------------------
        # Lens detector
        # ----------------------------------------------------

        self.lens_detector = LensDetector()

        print("FaceAware Pipeline initialized.")

    # ========================================================
    # Main processing
    # ========================================================

    def process(
        self,
        image,
        skin_strength=0.0,
        eye_bag_strength=0.0,
        eye_strength=0.0,
        iris_strength=0.0,
        teeth_strength=0.0,
        glasses_strength=0.0,
        wrinkle_strength=0.0
    ):

        if image is None:
            raise ValueError("Input image is None.")

        # Ensure uint8 image
        if image.dtype != np.uint8:
            image = np.clip(
                image,
                0,
                255
            ).astype(np.uint8)

        result = image.copy()

        # ====================================================
        # STEP 1
        # Face parsing
        # ====================================================

        parsing = self.face_parser.predict(image)

        if parsing is None:
            print("Warning: Face parsing failed.")
            return result

        # ====================================================
        # STEP 2
        # Extract semantic regions
        # ====================================================

        region_extractor = RegionExtractor(
            parsing
        )

        skin_mask = region_extractor.get_skin()

        glasses_mask = region_extractor.get_glasses()

        mouth_mask = region_extractor.get_mouth()

        # ====================================================
        # STEP 3
        # Facial landmarks
        # ====================================================

        landmarks = self.landmarks.get_points(
            image
        )

        if landmarks is None:
            print("Warning: No facial landmarks detected.")
            return result

        # ====================================================
        # STEP 4
        # Landmark-based masks
        # ====================================================

        region_masks = RegionMasks(
            image.shape
        )

        # ----------------------------------------------------
        # Eye masks
        # ----------------------------------------------------

        left_eye_mask = region_masks.create_eye_mask(
            landmarks,
            LEFT_EYE
        )

        right_eye_mask = region_masks.create_eye_mask(
            landmarks,
            RIGHT_EYE
        )

        eye_mask = combine_masks(
            left_eye_mask,
            right_eye_mask
        )

        # ----------------------------------------------------
        # Iris masks
        # ----------------------------------------------------

        left_iris_mask = region_masks.create_iris_mask(
            landmarks,
            LEFT_IRIS
        )

        right_iris_mask = region_masks.create_iris_mask(
            landmarks,
            RIGHT_IRIS
        )

        iris_mask = combine_masks(
            left_iris_mask,
            right_iris_mask
        )

        # ----------------------------------------------------
        # Inner mouth
        # ----------------------------------------------------

        inner_mouth_mask = (
            region_masks.create_inner_mouth_mask(
                landmarks,
                INNER_MOUTH
            )
        )

        # ====================================================
        # STEP 5
        # Skin smoothing
        # ====================================================

        if skin_strength > 0:

            result = smooth_skin(
                result,
                skin_mask,
                strength=skin_strength
            )

        # ====================================================
        # STEP 6
        # Eye whitening
        # ====================================================

        if eye_strength > 0:

            sclera_mask = (
                self.eye_enhancer.create_sclera_mask(
                    eye_mask,
                    iris_mask
                )
            )

            result = self.eye_enhancer.whiten(
                result,
                sclera_mask,
                strength=eye_strength
            )

        # ====================================================
        # STEP 7
        # Iris enhancement
        # ====================================================

        if iris_strength > 0:

            result = self.iris_enhancer.enhance(
                result,
                iris_mask,
                strength=iris_strength
            )

        # ====================================================
        # STEP 8
        # Eye-bag reduction
        # ====================================================

        if eye_bag_strength > 0:

            eye_bag_generator = EyeBagMask(
                image.shape
            )

            left_eye_bag = eye_bag_generator.create(
                landmarks,
                LEFT_EYE
            )

            right_eye_bag = eye_bag_generator.create(
                landmarks,
                RIGHT_EYE
            )

            eye_bag_mask = combine_masks(
                left_eye_bag,
                right_eye_bag
            )

            result = self.eye_bag_enhancer.reduce(
                result,
                eye_bag_mask,
                strength=eye_bag_strength
            )

        # ====================================================
        # STEP 9
        # Teeth whitening
        # ====================================================

        if teeth_strength > 0:

            teeth_mask = self._create_teeth_mask(
                result,
                mouth_mask,
                inner_mouth_mask
            )

            result = self.teeth_enhancer.whiten(
                result,
                teeth_mask,
                strength=teeth_strength
            )

        # ====================================================
        # STEP 10
        # Glasses glare removal
        # ====================================================

        has_glasses = (
            glasses_mask is not None
            and cv2.countNonZero(glasses_mask) > 50
        )

        if glasses_strength > 0 and has_glasses:

            try:
                lens_mask = self.lens_detector.detect_from_points(
                    landmarks,
                    image.shape
                )
            except Exception as e:
                print(f"Warning: Lens detection failed: {e}")
                lens_mask = glasses_mask      # fallback: parser mask

            if lens_mask is not None and cv2.countNonZero(lens_mask) > 0:

                # glare is auto-detected inside remove_glare
                empty_glare = np.zeros(image.shape[:2], np.uint8)

                result = self.glasses_enhancer.remove_glare(
                    result,
                    empty_glare,
                    lens_mask=lens_mask,
                    strength=glasses_strength
                )

                # ====================================================
        # STEP 11
        # Wrinkle reduction
        # ====================================================

        if wrinkle_strength > 0:

            ellipse = lambda k: cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (k, k)
            )

            # Shrink skin mask so boundary edges are not detected
            safe_skin = cv2.erode(skin_mask, ellipse(9))

            # Exclude eyes, iris, inner mouth and glasses
            exclude = combine_masks(
                eye_mask,
                iris_mask,
                inner_mouth_mask
            )

            if glasses_mask is not None:
                exclude = cv2.bitwise_or(exclude, glasses_mask)

            exclude = cv2.dilate(exclude, ellipse(15), iterations=2)

            safe_skin = cv2.bitwise_and(
                safe_skin,
                cv2.bitwise_not(exclude)
            )

            wrinkle_mask = (
                self.wrinkle_enhancer.create_wrinkle_mask(
                    result,
                    safe_skin
                )
            )

            result = self.wrinkle_enhancer.reduce(
                result,
                wrinkle_mask,
                strength=wrinkle_strength
            )
        # ====================================================
        # STEP 12
        # Final output
        # ====================================================

        result = np.clip(
            result,
            0,
            255
        ).astype(np.uint8)

        return result

    # ========================================================
    # Teeth mask
    # ========================================================

    def _create_teeth_mask(
        self,
        image,
        mouth_mask,
        inner_mouth_mask
    ):

        # Start with inner mouth
        mask = inner_mouth_mask.copy()

        if cv2.countNonZero(mask) == 0:
            return mask

        # Restrict to parser mouth region
        if mouth_mask is not None:

            mask = cv2.bitwise_and(
                mask,
                mouth_mask
            )

        if cv2.countNonZero(mask) == 0:
            return mask

        # ----------------------------------------------------
        # HSV
        # ----------------------------------------------------

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        h, s, v = cv2.split(hsv)

        # ----------------------------------------------------
        # Teeth are generally bright and relatively
        # low in saturation.
        # ----------------------------------------------------

        bright = v > 100
        low_saturation = s < 100

        candidate = (
            bright
            & low_saturation
            & (mask > 0)
        )

        teeth_mask = (
            candidate.astype(np.uint8) * 255
        )

        # ----------------------------------------------------
        # Remove noise
        # ----------------------------------------------------

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        teeth_mask = cv2.morphologyEx(
            teeth_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        teeth_mask = cv2.morphologyEx(
            teeth_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # ----------------------------------------------------
        # Smooth boundary
        # ----------------------------------------------------

        teeth_mask = cv2.GaussianBlur(
            teeth_mask,
            (5, 5),
            0
        )

        return teeth_mask