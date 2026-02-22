# Retinal OCT Disease Classification using DenseNet121

A deep learning project for automated classification of **Optical Coherence Tomography (OCT)** retinal scans into four diagnostic categories using **DenseNet121** with transfer learning.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Workflow](#workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Challenges & Solutions](#challenges--solutions)
- [Why DenseNet121?](#why-densenet121)
- [Tech Stack](#tech-stack)

---

## Problem Statement

Retinal diseases are a leading cause of vision impairment worldwide. Early and accurate diagnosis through OCT imaging is critical for timely treatment. This project automates the classification of OCT scans into four categories:

| Class | Description |
|-------|-------------|
| **CNV** | Choroidal Neovascularization — abnormal blood vessel growth beneath the retina |
| **DME** | Diabetic Macular Edema — fluid leakage causing retinal swelling |
| **DRUSEN** | Drusen deposits — early indicator of age-related macular degeneration |
| **NORMAL** | Healthy retina with no pathological findings |

---

## Dataset

The project uses the **Retinal OCT** dataset containing grayscale OCT scan images organized into class-specific folders.

### Dataset Statistics

| Split | Total Images | Per Class |
|-------|-------------|-----------|
| **Train** | 9,200 | 2,300 |
| **Validation** | 1,400 | 350 |
| **Test** | 1,400 | 350 |

> **Note:** The original source dataset contained 108,309 training images with severe class imbalance (NORMAL: 51,140 vs DRUSEN: 8,616). A balanced subset was created for training.

### Original Class Distribution (Before Balancing)

| Class | Count | Percentage |
|-------|-------|------------|
| NORMAL | 51,140 | 47.2% |
| CNV | 37,205 | 34.4% |
| DME | 11,348 | 10.5% |
| DRUSEN | 8,616 | 8.0% |

### Image Properties

- **Format:** JPEG/PNG grayscale images (R=G=B channels identical)
- **Original sizes:** Varying (mostly 496×512 and 496×768)
- **Resized to:** 224×224×3 for model input
- **Mean pixel intensities:** CNV (48.4) > NORMAL (46.2) > DME (44.1) > DRUSEN (42.2)

---

## Project Structure

```
DIP/
├── README.md                                  # This file
├── requirements.txt                           # Python dependencies
│
├── RetinalOCT_Dataset/                        # Dataset directory
│   ├── train/                                 # Training images
│   │   ├── CNV/
│   │   ├── DME/
│   │   ├── DRUSEN/
│   │   └── NORMAL/
│   ├── val/                                   # Validation images
│   │   ├── CNV/
│   │   ├── DME/
│   │   ├── DRUSEN/
│   │   └── NORMAL/
│   └── test/                                  # Test images
│       ├── CNV/
│       ├── DME/
│       ├── DRUSEN/
│       └── NORMAL/
│
├── EDA_Retinal.ipynb                          # Exploratory Data Analysis (augmentation-based)
├── densenet_train_eda.ipynb                   # EDA + DenseNet121 training (undersampling-based)
├── densenet121_retinalOCT.ipynb               # Main training notebook (two-phase transfer learning)
├── final_model.ipynb                          # Final model training on full augmented dataset
├── Model.ipynb                                # Initial from-scratch DenseNet121 attempt
├── phase_metrics.ipynb                        # Phase-wise evaluation metrics
│
├── densenet121_retinalOCT_phase1_best.keras   # Best Phase 1 model checkpoint
├── densenet121_retinalOCT_phase2_best.keras   # Best Phase 2 model checkpoint (final model)
├── test_results.json                          # Test set evaluation results
│
├── confusion_matrix.png                       # Test confusion matrix
├── confusion_matrix_normalized.png            # Normalized confusion matrix
├── phase1_training.png                        # Phase 1 training curves
└── phase2_training.png                        # Phase 2 training curves
```

---

## Architecture

### Model: DenseNet121 with Transfer Learning

```
Input (224 × 224 × 3)
    │
    ▼
DenseNet Preprocessing (normalization)
    │
    ▼
DenseNet121 Base (pretrained on ImageNet, 7,037,504 params)
    │
    ▼
Global Average Pooling 2D
    │
    ▼
Dropout (0.5)
    │
    ▼
Dense (4 units, softmax activation)
    │
    ▼
Output: [CNV, DME, DRUSEN, NORMAL] probabilities
```

### Model Summary

| Component | Parameters |
|-----------|-----------|
| DenseNet121 Base | 7,037,504 |
| Classifier Head | 4,100 |
| **Total** | **7,041,604 (26.86 MB)** |

---

## Workflow

### Stage 1: Exploratory Data Analysis

Performed in `EDA_Retinal.ipynb` and `densenet_train_eda.ipynb`:

- Class distribution visualization (before and after balancing)
- Image dimension analysis (scatter plots)
- Pixel intensity histograms (RGB and grayscale per class)
- Mean intensity comparison across classes
- Grayscale image grid visualization
- Intensity distribution box plots
- Data augmentation experiments (rotation, flips, zoom, contrast, brightness)

### Stage 2: Two-Phase Transfer Learning

Performed in `densenet121_retinalOCT.ipynb`:

#### Phase 1 — Feature Extraction (Frozen Base)

- **Base model:** Frozen (non-trainable)
- **Trainable params:** 4,100 (classifier head only)
- **Optimizer:** Adam (lr=3e-4)
- **Epochs:** 25
- **Purpose:** Train the classifier head to map DenseNet features to OCT classes

#### Phase 2 — Fine-Tuning (Unfrozen Base)

- **Base model:** Fully trainable
- **Trainable params:** 7,041,604 (all layers)
- **Optimizer:** Adam (lr=1e-5, 10× lower than Phase 1)
- **Epochs:** 25 (early stopped at epoch 14)
- **Purpose:** Fine-tune the entire network for OCT-specific features

### Stage 3: Evaluation

- Test set predictions with best Phase 2 checkpoint
- Classification report (precision, recall, F1-score per class)
- Confusion matrix (absolute and normalized)
- Results saved to `test_results.json`

### Training Callbacks

| Callback | Configuration |
|----------|--------------|
| **ModelCheckpoint** | Save best model based on `val_loss` |
| **EarlyStopping** | Patience=6, restore best weights |
| **ReduceLROnPlateau** | Factor=0.5, patience=3, min_lr=1e-6 (Phase 1) / 1e-7 (Phase 2) |
| **PlotTraining** | Custom callback to save training curves after each epoch |

---

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended, CPU training is possible but slow)

### Setup

```bash
# Clone or download the project
cd DIP

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
numpy
matplotlib
seaborn
scikit-learn
tensorflow>=2.17.0
keras
pillow
```

---

## Usage

### Training

1. Place the dataset in `RetinalOCT_Dataset/` with `train/`, `val/`, and `test/` subdirectories, each containing class folders (`CNV/`, `DME/`, `DRUSEN/`, `NORMAL/`).

2. Open and run `densenet121_retinalOCT.ipynb` sequentially:
   - Cells 1–6: Data loading and preprocessing
   - Cells 7–9: Model definition and callbacks
   - Cells 10–11: Phase 1 training (feature extraction)
   - Cells 12–13: Phase 2 training (fine-tuning)
   - Cells 14–18: Evaluation and results

### Evaluation Only

If you have a trained model checkpoint:

```python
from tensorflow import keras
import numpy as np

model = keras.models.load_model('densenet121_retinalOCT_phase2_best.keras')

# Predict on a single image
img = keras.utils.load_img('path/to/image.jpeg', target_size=(224, 224))
img_array = keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions)

print(f"Predicted: {predicted_class} ({confidence:.2%})")
```

---

## Results

### Overall Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **92.29%** |
| **Test Loss** | **0.2429** |
| **Macro F1-Score** | **92.28%** |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| CNV | 91.34% | 93.43% | 92.37% | 350 |
| DME | 95.74% | 90.00% | 92.78% | 350 |
| DRUSEN | 91.01% | 89.71% | 90.36% | 350 |
| NORMAL | 91.30% | 96.00% | 93.59% | 350 |

### Confusion Matrix

|  | Predicted CNV | Predicted DME | Predicted DRUSEN | Predicted NORMAL |
|--|:---:|:---:|:---:|:---:|
| **Actual CNV** | **327** | 6 | 16 | 1 |
| **Actual DME** | 13 | **315** | 5 | 17 |
| **Actual DRUSEN** | 18 | 4 | **314** | 14 |
| **Actual NORMAL** | 0 | 4 | 10 | **336** |

### Training Progression

| Phase | Train Accuracy | Validation Accuracy | Epochs |
|-------|---------------|-------------------|--------|
| Phase 1 (frozen) | 72.65% | 79.14% | 25 |
| Phase 2 (fine-tuned) | 95.28% | 92.21% | 14 (early stopped) |

### Comparison: From-Scratch vs Transfer Learning

| Approach | Train Acc | Val/Test Acc | Overfitting Gap |
|----------|-----------|-------------|-----------------|
| From-scratch (Model.ipynb) | 89.4% | 53.4% | **36%** |
| Transfer learning (final) | 95.3% | **92.3%** | **3%** |

---

## Challenges & Solutions

### 1. Training from Scratch Failed

**Problem:** Custom DenseNet121 trained from scratch achieved only 53.4% validation accuracy with severe overfitting (36% train-val gap) on 6,000 images.

**Solution:** Switched to **ImageNet-pretrained weights** with two-phase transfer learning, reducing the gap to 3%.

### 2. Severe Class Imbalance

**Problem:** Original dataset had 6× more NORMAL images than DRUSEN (51,140 vs 8,616).

**Approaches tried:**
- Data augmentation (rotation, flips, zoom, contrast) — crashed on corrupt files
- Undersampling to smallest class (8,616/class) — loses data
- **Final approach:** Curated balanced subset (2,300/class) with class weights

### 3. Limited GPU Availability

**Problem:** Local machine had no GPU; CPU training took ~7 hours.

**Solution:** Used remote server with NVIDIA RTX 3090 for faster iterations (~40s/epoch vs ~270s/epoch).

### 4. GPU Memory Constraints

**Problem:** Even with RTX 3090, `CUDA_ERROR_OUT_OF_MEMORY` occurred.

**Solution:** Used `tf.data` pipelines with streaming (`batch`, `prefetch`, `map`) instead of loading all images into memory. Kept batch size at 32.

### 5. Fine-Tuning Instability

**Problem:** High learning rate during fine-tuning destroyed pretrained features.

**Solution:** Used 10× lower learning rate (1e-5 vs 3e-4) for Phase 2, with `ReduceLROnPlateau` and `EarlyStopping` callbacks.

---

## Why DenseNet121?

| Factor | Advantage |
|--------|-----------|
| **Feature reuse** | Dense connections (each layer receives input from all preceding layers) maximize feature reuse — critical with limited medical data |
| **Parameter efficiency** | 7M params vs ResNet-50's 25M — lighter, faster, less overfitting |
| **Gradient flow** | Short paths from loss to early layers enable stable fine-tuning |
| **Medical imaging proven** | DenseNet121 is the backbone of CheXNet (Stanford's chest X-ray model) — validated on grayscale medical images |
| **Compact size** | 121 layers balances depth with practicality for constrained hardware |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | TensorFlow 2.17+ / Keras |
| **Model** | DenseNet121 (ImageNet pretrained) |
| **Language** | Python 3.10+ |
| **Data Pipeline** | tf.data API |
| **Evaluation** | scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Image Processing** | Pillow (PIL) |
| **Hardware** | NVIDIA RTX 3090 (training), CPU (inference) |

---

## License

This project is developed for academic purposes as part of the V-Semester Digital Image Processing (DIP) coursework.

---

## Acknowledgments

- [Retinal OCT Dataset](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) — Kermany et al., 2018
- DenseNet architecture — Huang et al., "Densely Connected Convolutional Networks" (CVPR 2017)
- CheXNet — Rajpurkar et al., Stanford ML Group
