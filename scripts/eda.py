"""
Exploratory data analysis for the small, pre-resized APTOS dataset
(class-folder format: gaussian_filtered_images/{No_DR, Mild, ...}).
Writes a class-distribution plot and a sample image grid to outputs/eda/.

Usage:
    python scripts/eda.py --data_dir data/raw/aptos_small/gaussian_filtered_images
"""

import argparse
import os
import cv2
import matplotlib.pyplot as plt

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
FOLDER_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]


def gather_samples(data_dir):
    samples = []
    for label, folder in enumerate(FOLDER_NAMES):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                samples.append((os.path.join(folder_path, fname), label))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", default="outputs/eda")
    parser.add_argument("--samples_per_class", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    samples = gather_samples(args.data_dir)
    if not samples:
        print(f"No images found under {args.data_dir}. Check the folder names match "
              f"{FOLDER_NAMES} and the path is correct.")
        return

    labels = [s[1] for s in samples]
    counts = [labels.count(c) for c in range(5)]
    print(f"Found {len(samples)} images. Class counts:",
          dict(zip(CLASS_NAMES, counts)))

    # --- Class distribution ---
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(CLASS_NAMES, counts, color="#C4661F")
    plt.title(f"APTOS (small) — class distribution ({len(samples)} images)")
    plt.ylabel("Image count")
    plt.xticks(rotation=20)
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), str(int(b.get_height())),
                  ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    dist_path = os.path.join(args.out_dir, "aptos_small_class_distribution.png")
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"Saved class distribution plot to {dist_path}")

    # --- Sample grid, one row per class ---
    import random
    random.seed(42)
    fig, axes = plt.subplots(5, args.samples_per_class,
                              figsize=(args.samples_per_class * 2.2, 5 * 2.2))
    for cls in range(5):
        cls_samples = [s for s in samples if s[1] == cls]
        random.shuffle(cls_samples)
        chosen = cls_samples[:args.samples_per_class]
        for col in range(args.samples_per_class):
            ax = axes[cls, col]
            ax.axis("off")
            if col < len(chosen):
                img = cv2.cvtColor(cv2.imread(chosen[col][0]), cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            if col == 0:
                ax.set_ylabel(CLASS_NAMES[cls], fontsize=10)
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])
    plt.tight_layout()
    grid_path = os.path.join(args.out_dir, "aptos_small_sample_grid.png")
    plt.savefig(grid_path, dpi=150)
    plt.close()
    print(f"Saved sample image grid to {grid_path}")
    print("\nClass imbalance ratio (max/min):", round(max(counts) / min(counts), 2))


if __name__ == "__main__":
    main()
