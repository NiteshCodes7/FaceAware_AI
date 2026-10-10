---
title: FaceAware AI
emoji: 🙂
sdk: docker
app_port: 7860
---

FaceAware AI backend (FastAPI). Region-aware facial enhancement API.

- `GET /health`
- `POST /enhance` (multipart: `image` plus `skin_strength`, `eye_strength`, `iris_strength`, `eye_bag_strength`, `teeth_strength`, `glasses_strength`, `wrinkle_strength`, each 0.0 to 1.0)