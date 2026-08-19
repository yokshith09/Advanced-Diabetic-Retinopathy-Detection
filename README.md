# EarlyDR — Project Setup Guide (small-dataset scope)

Scoped down to the smallest dataset combination that still supports the full project:
the pre-resized APTOS 2019 set (already sorted into class folders, ~200-300 MB) as the
main training set, with IDRiD (~200 MB, IEEE Dataport) as an optional smaller add-on for
cross-dataset testing later. EyePACS is intentionally left out.

Run everything below **on your own machine or in Colab/Kaggle** — Kaggle and IEEE
Dataport aren't reachable from this sandbox, so downloading needs your own connection
and account.

---

## 0. Environment setup

```bash
python3 -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
GPU users: install the CUDA build of PyTorch from https://pytorch.org/get-started/locally/
first, then re-run the line above for the rest.

---

## 1. Get a Kaggle API key

1. https://www.kaggle.com/settings/account → "Create New Token" → downloads `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` (macOS/Linux) or `C:\Users\<you>\.kaggle\kaggle.json`.
3. `chmod 600 ~/.kaggle/kaggle.json`.

---

## 2. Download the dataset

**APTOS 2019, small pre-resized version** (main dataset, use this one):
- Link: https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered
```bash
bash scripts/download_aptos_small.sh
```
Already resized to 224×224 and Gaussian-filtered, sorted into 5 class folders
(No_DR, Mild, Moderate, Severe, Proliferate_DR) — no manual preprocessing needed.

**IDRiD** (optional, small — for later cross-dataset testing):
- Link: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
- Needs a free IEEE Dataport account (no CLI/API) — download the **"B. Disease Grading"**
  sub-folder only (~200 MB), unzip into `data/raw/idrid/`.

---

## 3. EDA

```bash
python scripts/eda.py --data_dir data/raw/aptos_small/gaussian_filtered_images
```
Writes a class-distribution bar chart and a sample image grid to `outputs/eda/` — use
these directly in your documentation's dataset section.

---

## 4. Train

One script, works for all three architecture tracks — just change `--model`:

```bash
python scripts/train.py --data_dir data/raw/aptos_small/gaussian_filtered_images --model resnet50
python scripts/train.py --data_dir data/raw/aptos_small/gaussian_filtered_images --model vit_base_patch16_224
python scripts/train.py --data_dir data/raw/aptos_small/gaussian_filtered_images --model mobilenetv3_small_100
```

What it does automatically:
- Stratified 70/15/15 train/val/test split (fixed seed = 42, so it's reproducible —
  everyone on the team gets the same split if they run it with the same seed)
- Class-weighted cross-entropy loss (targets the Mild/Moderate recall objective)
- Standard augmentation (flips, rotation, color jitter)
- Tracks **quadratic weighted kappa** (the standard DR metric) and per-class recall every epoch
- Saves the best checkpoint by validation QWK to `outputs/checkpoints/<model>_best.pt`
- Runs a final test-set evaluation and writes accuracy, QWK, per-class recall, and the
  confusion matrix to `outputs/checkpoints/<model>_results.json`

Useful flags: `--epochs` (default 15), `--batch_size` (default 32), `--lr` (default 1e-4),
`--image_size` (default 224). Any [timm](https://github.com/huggingface/pytorch-image-models)
model name works for `--model` — swap in `efficientnet_b0`, `deit_base_patch16_224`,
`mobilenetv2_100`, etc. if your track needs a different variant.

**Team coordination:** since the split is generated with a fixed seed inside the script,
all three of you get an identical train/val/test split automatically as long as you keep
`--seed 42` (the default) — no separate split-freezing step needed for this smaller setup.

---

## 5. Comparing your results

Put each model's `outputs/checkpoints/<model>_results.json` side by side — accuracy, QWK,
and per-class recall — for the Review 2 comparison table. Reference points from the
published literature (all on APTOS) to sanity-check against: ResNet50 ≈ 0.90 QWK,
DenseNet121 ≈ 0.91 QWK, EfficientNetB0 (tuned) ≈ 0.92 QWK, MobileNetV3+ordinal head ≈ 0.90
QWK. A realistic target band for a well-tuned model here is roughly 0.85-0.92 QWK — don't
be alarmed if your first run is lower before tuning (Milestone 2 in your roadmap doc).

---

## Folder structure

```
EarlyDR-Project/
├── README.md
├── requirements.txt
├── config.yaml
├── scripts/
│   ├── download_aptos_small.sh   ← the dataset to use
│   ├── eda.py
│   └── train.py                   ← one script, all 3 architecture tracks
├── data/raw/aptos_small/          ← downloaded data goes here
├── outputs/
│   ├── eda/                       ← class distribution + sample grid
│   └── checkpoints/               ← best model .pt + results.json per architecture
├── docs/                          ← Review-1 documentation + member roadmaps
└── ui/                             ← screening console + about page
```
