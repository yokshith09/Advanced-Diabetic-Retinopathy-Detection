"""
Trains a DR severity classifier on the small, pre-resized APTOS dataset
(class-folder format). Works for all three architecture tracks -- just
change --model. Reports quadratic weighted kappa (the standard metric in
the DR literature) and per-class recall every epoch, and saves the best
checkpoint by validation QWK.

Usage:
    python scripts/train.py --data_dir data/raw/aptos_small/gaussian_filtered_images --model resnet50
    python scripts/train.py --data_dir data/raw/aptos_small/gaussian_filtered_images --model vit_base_patch16_224
    python scripts/train.py --data_dir data/raw/aptos_small/gaussian_filtered_images --model mobilenetv3_small_100

Any model name from timm (https://github.com/huggingface/pytorch-image-models)
works -- these three map to the project's CNN / ViT / MobileNet tracks.
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, recall_score, confusion_matrix
from PIL import Image
import timm
from tqdm import tqdm

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
# The pre-resized Kaggle dataset uses these folder names, in class order 0-4
FOLDER_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]


class DRDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples  # list of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


def gather_samples(data_dir):
    samples = []
    for label, folder in enumerate(FOLDER_NAMES):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                samples.append((os.path.join(folder_path, fname), label))
    return samples


def build_transforms(image_size, train):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    per_class_recall = recall_score(all_labels, all_preds, average=None, labels=list(range(5)))
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(5)))
    return qwk, per_class_recall, acc, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                         help="Path to the class-folder dataset root (e.g. .../gaussian_filtered_images)")
    parser.add_argument("--model", default="resnet50",
                         help="timm model name, e.g. resnet50 / efficientnet_b0 / vit_base_patch16_224 / mobilenetv3_small_100")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="outputs/checkpoints")
    parser.add_argument("--resume_from", default=None, help="Path to best checkpoint to resume from")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Data ---
    samples = gather_samples(args.data_dir)
    if not samples:
        print(f"No images found under {args.data_dir}. Check the folder names match "
              f"{FOLDER_NAMES} and the path is correct.")
        return
    labels = [s[1] for s in samples]
    print(f"Found {len(samples)} images. Class counts:",
          {CLASS_NAMES[c]: labels.count(c) for c in range(5)})

    train_val, test = train_test_split(samples, test_size=0.15, stratify=labels, random_state=args.seed)
    train_labels = [s[1] for s in train_val]
    train, val = train_test_split(train_val, test_size=0.1765, stratify=train_labels, random_state=args.seed)
    # 0.1765 of the 85% remainder ~= 15% of the total -> 70/15/15 split overall
    print(f"Split -> train: {len(train)}, val: {len(val)}, test: {len(test)}")

    train_ds = DRDataset(train, build_transforms(args.image_size, train=True))
    val_ds = DRDataset(val, build_transforms(args.image_size, train=False))
    test_ds = DRDataset(test, build_transforms(args.image_size, train=False))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # --- Class weights, for the early-stage-sensitivity objective ---
    train_label_arr = np.array([s[1] for s in train])
    class_counts = np.array([np.sum(train_label_arr == c) for c in range(5)])
    class_weights = train_label_arr.shape[0] / (5 * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Class weights:", {CLASS_NAMES[c]: round(float(w), 3) for c, w in enumerate(class_weights)})

    # --- Model ---
    model = timm.create_model(args.model, pretrained=True, num_classes=5)
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Loading previous weights from {args.resume_from} to continue training...")
        model.load_state_dict(torch.load(args.resume_from, map_location=device))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_qwk = -1
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        scheduler.step()

        train_loss = running_loss / len(train_ds)
        qwk, per_class_recall, acc, _ = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}  val_acc={acc:.4f}  val_QWK={qwk:.4f}")
        print("  per-class recall:", {CLASS_NAMES[c]: round(float(r), 3) for c, r in enumerate(per_class_recall)})

        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": float(acc),
                         "val_qwk": float(qwk),
                         "val_recall": {CLASS_NAMES[c]: float(r) for c, r in enumerate(per_class_recall)}})

        if qwk > best_qwk:
            best_qwk = qwk
            ckpt_path = os.path.join(args.out_dir, f"{args.model}_best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> new best val QWK ({qwk:.4f}), saved to {ckpt_path}")

    # --- Final test evaluation using the best checkpoint ---
    model.load_state_dict(torch.load(os.path.join(args.out_dir, f"{args.model}_best.pt")))
    test_qwk, test_recall, test_acc, test_cm = evaluate(model, test_loader, device)
    print("\n=== Final test set results ===")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test QWK: {test_qwk:.4f}")
    print("Per-class recall:", {CLASS_NAMES[c]: round(float(r), 3) for c, r in enumerate(test_recall)})
    print("Confusion matrix:\n", test_cm)

    results = {
        "model": args.model,
        "test_accuracy": float(test_acc),
        "test_qwk": float(test_qwk),
        "test_recall": {CLASS_NAMES[c]: float(r) for c, r in enumerate(test_recall)},
        "confusion_matrix": test_cm.tolist(),
        "history": history,
    }
    results_path = os.path.join(args.out_dir, f"{args.model}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {results_path}")


if __name__ == "__main__":
    main()
