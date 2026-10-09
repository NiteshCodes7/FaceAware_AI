import cv2
import numpy as np

image = cv2.imread("test4.jpg")
h, w = image.shape[:2]

lens = np.zeros((h, w), np.uint8)

# (center_x, center_y), (semi_axis_x, semi_axis_y)
cv2.ellipse(lens, (750, 672), (190, 205), 0, 0, 360, 255, -1)    # left lens
cv2.ellipse(lens, (1245, 805), (205, 205), 0, 0, 360, 255, -1)   # right lens

cv2.imwrite("lens_mask.png", lens)

vis = image.copy()
vis[lens > 127] = (vis[lens > 127] * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)
cv2.imwrite("lens_mask_debug.jpg", vis)