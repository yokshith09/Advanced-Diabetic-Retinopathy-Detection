import argparse
import os
import json
import torch
import timm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from train import DRDataset, build_transforms, gather_samples, evaluate, CLASS_NAMES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw/aptos_small/gaussian_filtered_images")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--weights_path", default="outputs/checkpoints/resnet50_best.pt")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="outputs/checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Recreate exactly the same test split
    samples = gather_samples(args.data_dir)
    labels = [s[1] for s in samples]
    train_val, test = train_test_split(samples, test_size=0.15, stratify=labels, random_state=args.seed)
    
    test_ds = DRDataset(test, build_transforms(args.image_size, train=False))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Load model
    model = timm.create_model(args.model, pretrained=False, num_classes=5)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model = model.to(device)

    print("Running evaluation on the test set...")
    test_qwk, test_recall, test_acc, test_cm = evaluate(model, test_loader, device)

    print("\n=== Final test set results ===")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test QWK: {test_qwk:.4f}")
    print("Per-class recall:", {CLASS_NAMES[c]: round(float(r), 3) for c, r in enumerate(test_recall)})

    results = {
        "model": args.model,
        "test_accuracy": float(test_acc),
        "test_qwk": float(test_qwk),
        "test_recall": {CLASS_NAMES[c]: float(r) for c, r in enumerate(test_recall)},
        "confusion_matrix": test_cm.tolist(),
        "history": [], # Skipping history since we didn't train in this script
    }
    
    results_path = os.path.join(args.out_dir, f"{args.model}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {results_path} successfully so you can use compare_results.py!")

if __name__ == "__main__":
    main()
