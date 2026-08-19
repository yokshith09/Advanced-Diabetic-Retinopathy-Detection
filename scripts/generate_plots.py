import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

def main():
    results_dir = "outputs/checkpoints"
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    # Models to plot
    models = {
        "resnet50": "ResNet50 (CNN Baseline)",
        "vit_base_patch16_224": "ViT Base (Transformer)",
        "mobilenetv3_small_100": "MobileNetV3 (Edge)"
    }
    
    # Colors
    colors = {
        "resnet50": "#e74c3c",      # Red-ish
        "vit_base_patch16_224": "#3498db",  # Blue-ish
        "mobilenetv3_small_100": "#2ecc71" # Green-ish
    }
    
    data = {}
    for m_id, label in models.items():
        path = os.path.join(results_dir, f"{m_id}_results.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                data[m_id] = json.load(f)
        else:
            print(f"Warning: {path} not found.")

    if not data:
        print("No results files found. Exiting.")
        return

    # Set seaborn style for nicer graphs
    sns.set_theme(style="whitegrid")
    
    # 1. Plot Convergence Curves
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: Training Loss
    plt.subplot(1, 2, 1)
    for m_id, label in models.items():
        if m_id in data:
            history = data[m_id]["history"]
            epochs = [h["epoch"] for h in history]
            loss = [h["train_loss"] for h in history]
            plt.plot(epochs, loss, marker='o', linewidth=2.5, color=colors[m_id], label=label)
    plt.title("Training Loss Convergence", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    
    # Subplot 2: Validation QWK
    plt.subplot(1, 2, 2)
    for m_id, label in models.items():
        if m_id in data:
            history = data[m_id]["history"]
            epochs = [h["epoch"] for h in history]
            qwk = [h["val_qwk"] for h in history]
            plt.plot(epochs, qwk, marker='s', linewidth=2.5, color=colors[m_id], label=label)
    plt.title("Validation Quadratic Weighted Kappa (QWK)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("QWK (Validation)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence_plots.png"), dpi=300)
    plt.close()
    print("Saved convergence_plots.png")

    # 2. Plot Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (m_id, label) in enumerate(models.items()):
        if m_id in data and "confusion_matrix" in data[m_id]:
            cm = np.array(data[m_id]["confusion_matrix"])
            
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False,
                        annot_kws={"size": 12, "weight": "bold"})
            
            axes[i].set_title(f"{label}\n(Test QWK: {data[m_id]['test_qwk']:.4f})", fontsize=13, fontweight='bold', pad=10)
            axes[i].set_xlabel("Predicted Grade", fontsize=11)
            if i == 0:
                axes[i].set_ylabel("True Grade", fontsize=11)
            else:
                axes[i].set_ylabel("")
            
            # Rotate labels for better readability
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=30, ha='right')
            axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0)
        else:
            axes[i].text(0.5, 0.5, f"No CM for {label}", ha='center', va='center')
            axes[i].axis('off')
            
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrices.png"), dpi=300)
    plt.close()
    print("Saved confusion_matrices.png")

if __name__ == "__main__":
    main()
