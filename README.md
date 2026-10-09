# Face Parser Models

This directory contains the pretrained and fine-tuned models used by the
FaceAware AI facial region parsing pipeline.

## Model Architecture

FaceAware AI uses **BiSeNet (Bilateral Segmentation Network)** for semantic
face parsing.

The network uses:

- **BiSeNet** as the segmentation architecture
- **ResNet-18** as the backbone
- **19 semantic classes** corresponding to facial and non-facial regions

The parser converts an input face image into a pixel-level segmentation mask.
Each pixel is assigned to one of the 19 predefined classes.

---

## Available Models

### 1. `resnet18.pt`

`resnet18.pt` is the initial pretrained checkpoint used as the starting point
for training.

It is loaded into the BiSeNet architecture with a ResNet-18 backbone and then
fine-tuned using the CelebAMask-HQ dataset prepared for this project.

This model is therefore used as the **initial pretrained model**, rather than
the final FaceAware AI face parser.

### 2. `training/bisenet_celebamask_best.pt`

This is the **fine-tuned FaceAware AI face parsing model**.

It was obtained by fine-tuning the initial `resnet18.pt` checkpoint on a
19-class CelebAMask-HQ dataset.

This is the model currently used by the FaceAware AI backend.

---

## Model Comparison

| Model | Purpose | Training |
|---|---|---|
| `resnet18.pt` | Initial pretrained checkpoint | Pretrained |
| `bisenet_celebamask_best.pt` | Final face parser | Fine-tuned on CelebAMask-HQ |

The final model should be used for inference in the FaceAware AI pipeline.

---

## Face Parsing Classes

The model predicts 19 classes:

| ID | Class |
|---:|---|
| 0 | Background |
| 1 | Skin |
| 2 | Nose |
| 3 | Eye Glasses |
| 4 | Left Eye |
| 5 | Right Eye |
| 6 | Left Eyebrow |
| 7 | Right Eyebrow |
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

These class IDs are important because the enhancement pipeline uses them to
extract specific facial regions.

For example:

```text
Class 1  → Skin
Class 3  → Glasses
Class 4/5 → Eyes
Class 10/11/12 → Mouth/Lips
Class 13 → Hair
```

This directory contains the pretrained and fine-tuned models used by the FaceAware AI facial region parsing pipeline.

## Dataset

The model was fine-tuned using a processed subset of the **CelebAMask-HQ** dataset.

For the current experiment:

```text
Total images : 1000

Training  : 800
Validation: 100
Testing   : 100
```

The original CelebAMask-HQ annotations were converted into a single 19-class segmentation mask matching the class IDs used by FaceAware AI.

---

## Training Configuration

The final model was fine-tuned using:

```text
Architecture : BiSeNet
Backbone     : ResNet-18
Classes      : 19
Input Size   : 512 × 512
Batch Size   : 2
Optimizer    : AdamW
Loss         : Cross Entropy Loss
Device       : CPU
```

The model was not trained completely from scratch.

The training process started from:

```text
resnet18.pt
```

and progressively fine-tuned the BiSeNet model using the prepared CelebAMask-HQ dataset.

---

## Final Evaluation

The best checkpoint was selected according to validation mIoU.

Final evaluation on the held-out test set:

```text
Pixel Accuracy : 94.49%
Mean IoU       : 73.17%
Test Loss      : 0.1674
```

### Per-Class IoU

| ID | Class         | IoU    |
|----|---------------|--------|
| 0  | Background    | 0.9231 |
| 1  | Skin          | 0.9248 |
| 2  | Nose          | 0.8742 |
| 3  | Eye Glasses   | 0.7376 |
| 4  | Left Eye      | 0.7957 |
| 5  | Right Eye     | 0.8045 |
| 6  | Left Eyebrow  | 0.7636 |
| 7  | Right Eyebrow | 0.7630 |
| 8  | Left Ear      | 0.8117 |
| 9  | Right Ear     | 0.7610 |
| 10 | Mouth         | 0.7821 |
| 11 | Upper Lip     | 0.7753 |
| 12 | Lower Lip     | 0.8124 |
| 13 | Hair          | 0.9073 |
| 14 | Hat           | 0.5519 |
| 15 | Earring       | 0.4249 |
| 16 | Neck Lower    | 0.0000 |
| 17 | Neck          | 0.8174 |
| 18 | Cloth         | 0.6727 |

---

## Why Fine-Tuning Was Required

The initial pretrained checkpoint did not provide sufficiently good 19-class segmentation for the target FaceAware AI pipeline.

The baseline evaluation was:

```text
Pixel Accuracy : 53.35%
Mean IoU       : 17.75%
```

After fine-tuning on the prepared CelebAMask-HQ data:

```text
Pixel Accuracy : 94.49%
Mean IoU       : 73.17%
```

This significantly improved the segmentation of facial regions such as:

- Skin
- Nose
- Eyes
- Eyebrows
- Lips
- Hair
- Glasses
- Ears
- Neck

The improvement is important because the enhancement modules depend on accurate region masks.

---

## Model Loading

The parser is loaded by:

```python
from segmentation.parser import FaceParser

parser = FaceParser(
    "models/face_parser/training/bisenet_celebamask_best.pt"
)
```

The parser internally creates:

```python
BiSeNet(
    num_classes=19,
    backbone_name="resnet18"
)
```

and loads the checkpoint weights.

---

## Inference

For an input image:

```python
segmentation_mask = parser.predict(image)
```

The returned mask has the same height and width as the original image.

Example:

```text
Input Image
     │
     ▼
BiSeNet + ResNet-18
     │
     ▼
19-Class Segmentation Mask
     │
     ├── Skin
     ├── Eyes
     ├── Glasses
     ├── Lips
     ├── Hair
     ├── Nose
     └── Other regions
```

The `RegionExtractor` then converts these class IDs into individual binary masks used by the enhancement modules.

---

## How the Model Is Used in FaceAware AI

The face parser is one component of the complete pipeline:

```text
Input Image
     │
     ▼
Face Parsing
(BiSeNet + ResNet-18)
     │
     ▼
19-Class Segmentation
     │
     ▼
Region Extraction
     │
     ├── Skin Mask
     ├── Eye Mask
     ├── Iris Mask
     ├── Teeth Mask
     ├── Glasses/Lens Mask
     └── Wrinkle/Skin Mask
     │
     ▼
Region-Specific Enhancement
     │
     ▼
Strength-Controlled Blending
     │
     ▼
Enhanced Image
```

The segmentation model does **not directly enhance the image**.

Its job is to identify *where* different facial regions are located.

The enhancement algorithms then operate only on the relevant regions.

---

## Model Files

Expected directory structure:

```text
models/
└── face_parser/
    ├── README.md
    ├── resnet18.pt
    └── training/
        └── bisenet_celebamask_best.pt
```

### Recommended Model

For FaceAware AI inference:

```text
training/bisenet_celebamask_best.pt
```

should be used.

The `resnet18.pt` file is retained as the initial pretrained checkpoint from which the final model was fine-tuned.

---

## Important Note

The final model is a **fine-tuned BiSeNet face parsing model**, not a BiSeNet model trained completely from scratch.

The training process was:

```text
Pretrained Checkpoint
        │
        ▼
    resnet18.pt
        │
        ▼
BiSeNet + ResNet-18
        │
        ▼
Fine-tuning on
CelebAMask-HQ
        │
        ▼
bisenet_celebamask_best.pt
```

This approach was chosen because training a segmentation network from scratch on CPU with a relatively small experimental dataset would require significantly more computational resources and training time.
"""
