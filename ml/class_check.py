import sys

import cv2
import numpy as np

from segmentation.parser import FaceParser

image_path = "test.jpeg"
model_path = "../models/face_parser/training/bisenet_celebamask_best.pt"
img = cv2.imread(image_path)
pred = FaceParser(model_path).predict(img)

if pred.shape != img.shape[:2]:
    pred = cv2.resize(
        pred.astype(np.uint8),
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

color = cv2.applyColorMap((pred * 13).astype(np.uint8), cv2.COLORMAP_JET)
vis = cv2.addWeighted(img, 0.5, color, 0.5, 0)
fs = max(img.shape[:2]) / 1500

for cid in np.unique(pred):
    if cid == 0:
        continue
    ys, xs = np.nonzero(pred == cid)
    cx, cy = int(xs.mean()), int(ys.mean())
    print(f"class {cid:2d}: {len(xs):8d} px  centroid=({cx},{cy})")
    cv2.putText(vis, str(cid), (cx - 10, cy + 10),cv2.FONT_HERSHEY_SIMPLEX, 1.2 * fs, (255, 255, 255), 3)

cv2.imwrite("class_ids.png", vis)
print("Saved class_ids.png")