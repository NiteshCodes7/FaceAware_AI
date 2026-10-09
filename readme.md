# FaceAware AI

**Region-Aware Intelligent Facial Image Enhancement and Restoration**

FaceAware AI is a computer-vision application that performs **region-aware facial image enhancement**. Instead of applying a single global beauty filter to the entire image, the system identifies individual facial regions and applies specialized enhancement techniques with **independent user-controlled strength**.

## Structure
- `ml/` FastAPI backend and enhancement pipeline
- `frontend/` Next.js UI

## Run
    # backend
    cd ml
    python -m venv venv && venv\Scripts\activate
    pip install -r requirements.txt
    python scripts/download_weights.py
    uvicorn main:app --port 8000

    # frontend
    cd frontend
    npm install
    npm run dev

## Models
- `v1-pretrained` tag / `pretrained` branch: pre-trained face parser
- Custom model: see the `v2-custom-model` release

## Features

- Face parsing with a fine-tuned **BiSeNet + ResNet-18** model
- 19-class facial region segmentation
- Facial landmark detection using **MediaPipe Face Mesh**
- Region-specific skin smoothing
- Eye whitening
- Iris enhancement
- Eye-bag reduction
- Teeth whitening
- Glasses/lens glare reduction
- Wrinkle reduction
- Independent enhancement strength controls
- Combined enhancement pipeline
- FastAPI backend
- Next.js frontend
- Downloadable enhanced image
- CPU-compatible inference

---

## System Architecture

```text
                         FaceAware AI
                              │
                              ▼
                       Image Upload
                              │
                              ▼
                     Next.js Frontend
                              │
                    POST /enhance
                              │
                              ▼
                       FastAPI Backend
                              │
                              ▼
                    FaceAware Pipeline
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
       Fine-tuned BiSeNet            MediaPipe Face Mesh
       19-class parsing               Facial landmarks
               │                             │
               └──────────────┬──────────────┘
                              ▼
                       Region Masks
                              │
          ┌───────────┬───────┼────────┬───────────┐
          ▼           ▼       ▼        ▼           ▼
        Skin        Eyes     Iris    Teeth      Glasses
          │           │       │        │           │
          └───────────┴───────┴────────┴───────────┘
                              │
                         Enhancements
                              │
                              ▼
                    Strength-based Blending
                              │
                              ▼
                       Enhanced Image
                              │
                              ▼
                        JPEG Response
                              │
                              ▼
                     Next.js Preview
```

---

## Core Idea

Traditional image enhancement often applies the same operation to the entire image.

FaceAware AI instead performs:

```text
Input Image
     │
     ▼
Face Parsing
     │
     ├── Skin       → Skin smoothing
     ├── Eyes       → Eye whitening
     ├── Iris       → Iris enhancement
     ├── Teeth      → Teeth whitening
     ├── Glasses    → Glare reduction
     ├── Eye bags   → Eye-bag reduction
     └── Wrinkles   → Wrinkle reduction
```

Each enhancement is applied only to its corresponding facial region.

The final result is controlled using:

```text
Final = Original × (1 - strength) + Enhanced × strength
```

Therefore:

- `strength = 0` → original image
- `strength = 0.5` → 50% enhancement
- `strength = 1` → full enhancement

---

# Machine Learning Model

## BiSeNet Face Parser

FaceAware AI uses **BiSeNet with a ResNet-18 backbone** for semantic face parsing.

The model predicts 19 classes:

| ID | Region |
|---:|---|
| 0 | Background |
| 1 | Skin |
| 2 | Nose |
| 3 | Glasses |
| 4 | Left Eye |
| 5 | Right Eye |
| 6 | Left Brow |
| 7 | Right Brow |
| 8 | Left Ear |
| 9 | Right Ear |
| 10 | Mouth |
| 11 | Upper Lip |
| 12 | Lower Lip |
| 13 | Hair |
| 14 | Hat |
| 15 | Earring |
| 16 | Neck Lower |
| 17 | Neck |
| 18 | Cloth |

---

## Training

The model was **fine-tuned**, not trained completely from random initialization.

### Dataset

**CelebAMask-HQ**

For the current training experiment:

```text
Total images: 1,000

Training:    800
Validation: 100
Testing:     100
```

The original dataset masks were processed and combined into the 19-class segmentation format used by FaceAware AI.

### Training configuration

```text
Model:          BiSeNet
Backbone:       ResNet-18
Classes:        19
Input size:     512 × 512
Batch size:     2
Optimizer:      AdamW
Loss:           Cross Entropy Loss
Device:         CPU
```

### Final Test Results

Evaluation was performed on **100 held-out test images**.

```text
Test Loss:       0.1674
Pixel Accuracy:  94.49%
Mean IoU:        73.17%
```

### Per-Class IoU

| Class | Region | IoU |
|---:|---|---:|
| 0 | Background | 92.31% |
| 1 | Skin | 92.48% |
| 2 | Nose | 87.42% |
| 3 | Glasses | 73.76% |
| 4 | Left Eye | 79.57% |
| 5 | Right Eye | 80.45% |
| 6 | Left Brow | 76.36% |
| 7 | Right Brow | 76.30% |
| 8 | Left Ear | 81.17% |
| 9 | Right Ear | 76.10% |
| 10 | Mouth | 78.21% |
| 11 | Upper Lip | 77.53% |
| 12 | Lower Lip | 81.24% |
| 13 | Hair | 90.73% |
| 14 | Hat | 55.19% |
| 15 | Earring | 42.49% |
| 16 | Neck Lower | 0.00% |
| 17 | Neck | 81.74% |
| 18 | Cloth | 67.27% |

---

# Enhancement Modules

## 1. Skin Smoothing

Skin regions are extracted from the face-parsing mask.

A bilateral filter is applied to smooth skin while preserving important facial edges.

```text
Skin Mask
    ↓
Bilateral Filter
    ↓
Strength-controlled blending
```

---

## 2. Eye Whitening

The eye region is extracted using facial landmarks and iris information.

The sclera is estimated by removing the iris region from the eye mask, after which whitening is applied.

---

## 3. Iris Enhancement

MediaPipe iris landmarks are used to construct iris masks.

The iris region is enhanced independently from the surrounding eye.

---

## 4. Eye-Bag Reduction

Eye-bag masks are created around the lower-eye region using facial landmarks.

The detected region is processed and blended according to the selected strength.

---

## 5. Teeth Whitening

The mouth region is first identified.

Color information in HSV space is then used to identify likely visible teeth regions before applying whitening.

If teeth are not visible in an input image, this module naturally produces little or no visible change.

---

## 6. Glasses Glare Removal

The pipeline estimates lens regions using eye/iris geometry and detects glare within those regions.

Detected glare can then be reduced using image restoration/inpainting techniques.

The effect depends on the presence and intensity of actual glare in the input image.

---

## 7. Wrinkle Reduction

Wrinkle detection uses image-processing techniques including:

```text
Black-hat filtering
       ↓
Thresholding
       ↓
Morphological processing
       ↓
Wrinkle mask
       ↓
Inpainting
       ↓
Strength-controlled blending
```

---

# Facial Landmarks

FaceAware AI uses **MediaPipe Face Mesh** for detailed facial landmarks.

These landmarks are used to construct precise masks for:

- Eyes
- Iris
- Mouth
- Inner mouth
- Eye-bag regions

This allows enhancement operations to be aligned with the actual facial geometry.

---

# Project Structure

```text
FaceAware_AI/
│
├── models/
│   └── face_parser/
│       ├── resnet18.pt
│       └── training/
│           └── bisenet_celebamask_best.pt
│
├── datasets/
│   └── celebAMask/
│       ├── images/
│       ├── masks/
│       ├── split/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── raw/
│
└── ml/
    ├── api.py
    ├── pipeline.py
    ├── test_pipeline.py
    ├── evaluate_parser.py
    ├── train_parser.py
    ├── prepare_celebamask.py
    │
    ├── segmentation/
    │   ├── parser.py
    │   └── region_extractor.py
    │
    ├── landmarks/
    │   └── ...
    │
    └── enhancement/
        ├── skin.py
        ├── eye.py
        ├── iris.py
        ├── eye_bag.py
        ├── eye_bag_mask.py
        ├── teeth.py
        ├── glasses.py
        ├── lens_detector.py
        └── wrinkle.py
```

---

# Backend

The backend is implemented using **FastAPI**.

## API Endpoint

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model": "bisenet_celebamask_best.pt"
}
```

### Image Enhancement

```http
POST /enhance
```

The endpoint accepts:

```text
image
skin_strength
eye_strength
iris_strength
eye_bag_strength
teeth_strength
glasses_strength
wrinkle_strength
```

Strength values are between:

```text
0.0 - 1.0
```

The backend returns the enhanced image as:

```text
image/jpeg
```

---

# Running the Backend

Activate the virtual environment:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install fastapi uvicorn python-multipart
```

Start the server:

```powershell
python -m uvicorn api:app --reload
```

The backend runs at:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/health
```

---

# Frontend

The frontend is implemented using:

- Next.js
- React
- Tailwind CSS

It provides:

- Image upload
- Drag-and-drop support
- Original image preview
- Enhanced image preview
- Seven enhancement sliders
- Processing indicator
- Error handling
- Enhanced image download

The frontend sends the image and slider values to:

```text
POST /enhance
```

---

# Enhancement Controls

| Control | Description |
|---|---|
| Skin | Skin smoothing strength |
| Eyes | Eye whitening strength |
| Iris | Iris enhancement strength |
| Eye Bags | Eye-bag reduction strength |
| Teeth | Teeth whitening strength |
| Glasses | Glasses glare reduction strength |
| Wrinkles | Wrinkle reduction strength |

Example:

```text
Skin       50%
Eyes       40%
Iris       60%
Eye Bags   50%
Teeth      70%
Glasses    80%
Wrinkles   40%
```

---

# Technology Stack

## Machine Learning / Computer Vision

- Python
- PyTorch
- BiSeNet
- ResNet-18
- OpenCV
- NumPy
- MediaPipe
- Pillow

## Backend

- FastAPI
- Uvicorn
- Python

## Frontend

- Next.js
- React
- TypeScript
- Tailwind CSS

## Dataset

- CelebAMask-HQ

---

# Pipeline

The complete processing pipeline is:

```text
                    Input Image
                         │
                         ▼
                  Image Preprocessing
                         │
                         ▼
                Fine-tuned BiSeNet
                         │
                         ▼
                 19-Class Face Mask
                         │
                         ▼
                 Region Extraction
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
        Skin           Eyes           Iris
          │              │              │
          ▼              ▼              ▼
      Smoothing      Whitening      Enhancement
          │              │              │
          ├──────────────┼──────────────┤
          │              │              │
          ▼              ▼              ▼
      Eye Bags        Teeth          Glasses
          │              │              │
          ▼              ▼              ▼
       Reduce         Whiten       Remove Glare
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                      Wrinkles
                         │
                         ▼
                  Strength Blending
                         │
                         ▼
                  Enhanced Image
```

---

# Example Results

The system has been tested on real facial images containing:

- Facial skin
- Glasses
- Visible lens glare
- Eyes and iris
- Visible teeth

The pipeline successfully produces region-specific enhancement while preserving the overall facial structure and background.

---

# Model Evaluation

The original BiSeNet checkpoint was compared with the fine-tuned model.

### Before Fine-tuning

```text
Pixel Accuracy: 53.35%
Mean IoU:       17.75%
```

### After Fine-tuning

```text
Pixel Accuracy: 94.49%
Mean IoU:       73.17%
```

This demonstrates a substantial improvement in face parsing performance on the CelebAMask-HQ test split.

---

# Future Improvements

Potential future improvements include:

- Training with the complete CelebAMask-HQ dataset
- Training BiSeNet from scratch for comparison
- Increasing segmentation accuracy for rare classes
- Better glasses/lens detection
- More advanced wrinkle restoration
- Improved teeth segmentation
- GPU inference optimization
- Batch processing
- Before/after interactive comparison
- More advanced perceptual evaluation using PSNR, SSIM and LPIPS
- User studies for enhancement quality
- Deployment to a cloud GPU/server

---

# Research Evaluation

The system can be evaluated from two perspectives.

### Segmentation

```text
Pixel Accuracy
IoU
Mean IoU
Dice Score
Per-class IoU
```

### Image Enhancement

```text
PSNR
SSIM
LPIPS
Human Preference
```

The segmentation evaluation measures whether the model correctly identifies facial regions, while enhancement evaluation measures the quality of the resulting image.

---

# Important Note About the Current Model

The current model is **fine-tuned from an existing BiSeNet/ResNet-18 checkpoint**.

It is therefore more accurate to describe the training process as:

> **Fine-tuning a pretrained BiSeNet model on a 1,000-image CelebAMask-HQ subset**

rather than:

> Training BiSeNet from scratch.

---

# Project Goal

FaceAware AI aims to demonstrate that facial image enhancement can be performed in a **region-aware, controllable and selective manner**, rather than applying a single global transformation to the entire image.

The central concept is:

```text
Detect → Understand → Segment → Enhance → Blend
```

**FaceAware AI — Region-Aware Intelligent Facial Image Enhancement and Restoration.**