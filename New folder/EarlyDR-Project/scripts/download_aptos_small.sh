#!/bin/bash
# Downloads the small, pre-resized (224x224, Gaussian-filtered) APTOS dataset.
# ~200-300MB, already sorted into class folders -- no raw preprocessing needed.
# Prerequisite: ~/.kaggle/kaggle.json set up (see README.md step 1).
set -e

DEST="data/raw/aptos_small"
mkdir -p "$DEST"

echo "Downloading pre-resized APTOS 2019 (Gaussian filtered, 224x224)..."
kaggle datasets download -d sovitrath/diabetic-retinopathy-224x224-gaussian-filtered -p "$DEST"

echo "Unzipping..."
cd "$DEST"
unzip -q diabetic-retinopathy-224x224-gaussian-filtered.zip
rm diabetic-retinopathy-224x224-gaussian-filtered.zip

echo "Done. Class folders are under $DEST/gaussian_filtered_images/"
