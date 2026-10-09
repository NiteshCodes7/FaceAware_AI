import cv2
import numpy as np

from enhancement.glasses import GlassesEnhancer
from enhancement.lens_detector import LensDetector

IMAGE_PATH = "test5.jpg"

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise RuntimeError(f"Could not load image: {IMAGE_PATH}")

# Auto lens mask
lens_mask = LensDetector().detect(image)
cv2.imwrite("lens_mask.png", lens_mask)

# Debug overlay (check that green covers both full lenses)
vis = image.copy()
vis[lens_mask > 127] = (
    vis[lens_mask > 127] * 0.6 + np.array([0, 255, 0]) * 0.4
).astype(np.uint8)
cv2.imwrite("lens_mask_debug.jpg", vis)

# Glare mask is detected inside remove_glare, so pass an empty one
glare_mask = np.zeros(image.shape[:2], np.uint8)

enhancer = GlassesEnhancer()

for strength in [0.25, 0.5, 0.75, 1.0]:
    result = enhancer.remove_glare(
        image=image,
        glare_mask=glare_mask,
        lens_mask=lens_mask,
        strength=strength,
    )
    out = f"glasses_glare_removed_{int(strength * 100)}.jpg"
    cv2.imwrite(out, result)
    print("Generated:", out)