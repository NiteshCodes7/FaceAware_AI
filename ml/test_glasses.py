import cv2
import numpy as np

from enhancement.glasses import GlassesEnhancer 
from enhancement.lens_detector import LensDetector            

img = cv2.imread("test4.jpg")

lens = LensDetector(radius=0.5).detect(img)
out = GlassesEnhancer().remove_glare(img, np.zeros(img.shape[:2], np.uint8), lens, strength=1.0)

# overlay to check the lens mask is on the lenses
overlay = img.copy()
overlay[lens > 0] = (overlay[lens > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

cv2.imwrite("result4.png", np.hstack([img, overlay, out]))   # original | lens mask | fixed
print("saved result.png")