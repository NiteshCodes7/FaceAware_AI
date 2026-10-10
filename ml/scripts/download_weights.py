import os
import sys
import urllib.request

REPO = "NiteshCodes7/FaceAware_AI"
TAG = os.getenv("WEIGHTS_TAG", "v2-custom-model")
FILE = os.getenv("WEIGHTS_FILE", "bisenet_celebamask_best.pt")
DEST = os.getenv("WEIGHTS_DIR", "../models/face_parser/training")

os.makedirs(DEST, exist_ok=True)
out = os.path.join(DEST, FILE)

if os.path.exists(out):
    print("Already exists:", out)
    sys.exit(0)

url = f"https://github.com/{REPO}/releases/download/{TAG}/{FILE}"
print("Downloading", url)

try:
    urllib.request.urlretrieve(url, out)
except Exception as e:
    print(f"Download failed: {e}\nURL: {url}")
    sys.exit(1)

print("Saved:", out, os.path.getsize(out) // (1024 * 1024), "MB")