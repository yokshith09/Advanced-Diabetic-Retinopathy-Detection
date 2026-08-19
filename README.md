<div align="center">
  <h1>👁️ EarlyDR</h1>
  <p><b>Advanced Diabetic Retinopathy Detection using Deep Learning</b></p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
  [![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103&style=for-the-badge)](https://github.com/ellerbrock/open-source-badges/)
  [![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
</div>

<br/>

> **Note:** This project is completely open-source and currently under active development. Our present focus is primarily on experimenting with various deep learning architectures and improving our evaluation metrics.

---

## 📖 Table of Contents
- [Problem Statement](#-problem-statement)
- [Scope & Dataset Overview](#-scope--dataset-overview)
- [Project Highlights](#-project-highlights)
- [Getting Started](#-getting-started)
  - [Environment Setup](#0-environment-setup)
  - [Kaggle API Key](#1-get-a-kaggle-api-key)
  - [Download the Dataset](#2-download-the-dataset)
- [Usage](#-usage)
  - [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
  - [Model Training](#4-model-training)
  - [Comparing Results](#5-comparing-your-results)
- [Directory Structure](#-folder-structure)

---

## 🎯 Problem Statement

**Diabetic Retinopathy (DR)** is a leading cause of blindness worldwide, stemming from diabetes complications that damage the retina's blood vessels. Early detection and timely treatment are critical to preventing irreversible vision loss. However, manual screening by ophthalmologists is resource-intensive, time-consuming, and prone to human error, particularly in regions with limited access to specialized eye care.

The objective of this project is to automate the detection and grading of Diabetic Retinopathy from retinal fundus images using advanced deep learning techniques. By developing robust and efficient models (such as **CNNs** and **Vision Transformers**), we aim to provide an accessible and reliable screening tool that assists clinicians in diagnosing DR severity faster and more accurately.

---

## 📊 Scope & Dataset Overview

We have scoped down the project to the most efficient dataset combination that still supports full robust training:

- **Primary Training Set**: The pre-resized [APTOS 2019 dataset](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered) (already sorted into class folders, ~200-300 MB).
- **Secondary Testing Set**: [IDRiD dataset](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) (~200 MB, IEEE Dataport) serving as an optional smaller add-on for cross-dataset testing later.
- *Note: The massive EyePACS dataset is intentionally left out to prioritize rapid prototyping.*

> **Important:** Run all steps below **on your own machine or in Colab/Kaggle** — Kaggle and IEEE Dataport require authentication, so downloading needs your own connection and account.

---

## ✨ Project Highlights

- **Multi-Architecture Support**: Seamlessly train ResNet50, Vision Transformers (ViT), MobileNetV3, and more.
- **Automated Stratified Splits**: Guaranteed reproducible train/val/test splits across teams.
- **Class-Weighted Optimization**: Built-in cross-entropy weighting targets the challenging Mild/Moderate DR recall objective.
- **Metric Tracking**: Automatically evaluates **Quadratic Weighted Kappa (QWK)** and per-class recall.

---

## 🚀 Getting Started

### 0. Environment Setup

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate    # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> 💡 **GPU Users:** Install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) first, then re-run the pip install command.

### 1. Get a Kaggle API Key

1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings/account).
2. Click **"Create New Token"** to download `kaggle.json`.
3. Place it in the appropriate directory:
   - **macOS/Linux**: `~/.kaggle/kaggle.json` (Run `chmod 600 ~/.kaggle/kaggle.json` for security)
   - **Windows**: `C:\Users\<your-username>\.kaggle\kaggle.json`

### 2. Download the Dataset

**APTOS 2019 (Small pre-resized version)** — *Main dataset*  
This dataset is already resized to 224×224, Gaussian-filtered, and sorted into 5 class folders (`No_DR`, `Mild`, `Moderate`, `Severe`, `Proliferate_DR`).

```bash
bash scripts/download_aptos_small.sh
```

**IDRiD (Optional)** — *For cross-dataset testing*  
Requires a free IEEE Dataport account. Download the **"B. Disease Grading"** sub-folder only (~200 MB) and unzip it into `data/raw/idrid/`.

---

## 💻 Usage

### 3. Exploratory Data Analysis (EDA)

Visualize the class distribution and sample images to understand the dataset better.

```bash
python scripts/eda.py --data_dir data/raw/aptos_small/gaussian_filtered_images
```
*Outputs are saved to `outputs/eda/`, providing a clear view for your documentation.*

### 4. Model Training

Our unified script works for all three architecture tracks. Just swap out the `--model` parameter!

```bash
# ResNet50
python scripts/train.py --data_dir data/raw/aptos_small/gaussian_filtered_images --model resnet50

# Vision Transformer (ViT)
python scripts/train.py --data_dir data/raw/aptos_small/gaussian_filtered_images --model vit_base_patch16_224

# MobileNetV3
python scripts/train.py --data_dir data/raw/aptos_small/gaussian_filtered_images --model mobilenetv3_small_100
```

**What the script automates:**
- Stratified 70/15/15 split (fixed `seed=42` for team coordination).
- Class-weighted cross-entropy loss.
- Standard augmentations (flips, rotation, color jitter).
- Saves the best checkpoint (by validation QWK) to `outputs/checkpoints/<model>_best.pt`.
- Saves final results (Accuracy, QWK, Recall, Confusion Matrix) to `outputs/checkpoints/<model>_results.json`.

> **Pro Tip:** Use flags like `--epochs`, `--batch_size`, `--lr`, and `--image_size` to customize your training run. You can also use any model from the [timm](https://github.com/huggingface/pytorch-image-models) library.

### 5. Comparing Your Results

To evaluate performance, compare each model's `outputs/checkpoints/<model>_results.json`. 

**Literature Benchmarks (APTOS)**:
- ResNet50 ≈ 0.90 QWK
- DenseNet121 ≈ 0.91 QWK
- EfficientNetB0 (tuned) ≈ 0.92 QWK
- MobileNetV3+ordinal head ≈ 0.90 QWK

*A realistic target band for a well-tuned model is roughly 0.85-0.92 QWK.*

---

## 📂 Folder Structure

```text
EarlyDR-Project/
├── README.md
├── requirements.txt
├── config.yaml
├── scripts/
│   ├── download_aptos_small.sh   ← the dataset to use
│   ├── eda.py
│   └── train.py                  ← one script, all 3 architecture tracks
├── data/raw/aptos_small/         ← downloaded data goes here
├── outputs/
│   ├── eda/                      ← class distribution + sample grid
│   └── checkpoints/              ← best model .pt + results.json per architecture
├── docs/                         ← Review-1 documentation + member roadmaps
└── ui/                           ← screening console + about page
```

---

<div align="center">
  <b>Built with ❤️ by the EarlyDR Team</b>
</div>
