import argparse
import os
import pandas as pd
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from sklearn.metrics import cohen_kappa_score, recall_score, accuracy_score, confusion_matrix
import timm

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

def main():
    parser = argparse.ArgumentParser(description="Cross-dataset testing of trained model on IDRiD")
    parser.add_argument("--model", default="vit_base_patch16_224", help="timm model name")
    parser.add_argument("--weights_path", default="outputs/checkpoints/vit_base_patch16_224_best.pt", help="Path to best checkpoint")
    parser.add_argument("--idrid_dir", default="data/processed/idrid_224_gaussian", help="Preprocessed IDRiD images directory")
    parser.add_argument("--labels_csv", default="data/raw/idrid/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv", help="IDRiD labels CSV")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.labels_csv):
        print(f"Error: Labels file {args.labels_csv} not found.")
        return

    if not os.path.exists(args.idrid_dir):
        print(f"Error: Preprocessed directory {args.idrid_dir} not found. Run preprocess_idrid.py first.")
        return

    # Load labels
    df = pd.read_csv(args.labels_csv)
    # Strip whitespace from column names if present
    df.columns = df.columns.str.strip()
    
    # Map image name to file path and target label
    # IDRiD test labels have columns 'Image name' and 'Retinopathy grade'
    samples = []
    for _, row in df.iterrows():
        img_name = row['Image name'] + ".jpg"
        label = int(row['Retinopathy grade'])
        img_path = os.path.join(args.idrid_dir, img_name)
        if os.path.exists(img_path):
            samples.append((img_path, label))

    if not samples:
        print(f"No matching images found in {args.idrid_dir} matching {args.labels_csv}")
        return

    print(f"Found {len(samples)} IDRiD testing images with groundtruth labels.")

    # Load model
    model = timm.create_model(args.model, pretrained=False, num_classes=5)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_preds, all_labels = [], []
    with torch.no_grad():
        for path, label in samples:
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            output = model(tensor)
            pred = output.argmax(dim=1).item()
            all_preds.append(pred)
            all_labels.append(label)

    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    acc = accuracy_score(all_labels, all_preds)
    per_class_recall = recall_score(all_labels, all_preds, average=None, labels=list(range(5)))
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(5)))

    print(f"\n=== Cross-Dataset Results: Trained on APTOS -> Tested on IDRiD ({args.model}) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"QWK:      {qwk:.4f}")
    print("Per-class recall:")
    for c_idx, r in enumerate(per_class_recall):
        print(f"  - {CLASS_NAMES[c_idx]}: {r:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()
