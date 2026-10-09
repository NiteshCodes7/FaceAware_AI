import cv2
import numpy as np

image = cv2.imread("test4.jpg")
lens = cv2.imread("lens_mask.png", cv2.IMREAD_GRAYSCALE)
glare = cv2.imread("glare_mask.png", cv2.IMREAD_GRAYSCALE)

vis = image.copy()
vis[lens > 127] = (vis[lens > 127] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)    # lens = green
vis[glare > 127] = (vis[glare > 127] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)  # glare = red
cv2.imwrite("mask_debug.jpg", vis)