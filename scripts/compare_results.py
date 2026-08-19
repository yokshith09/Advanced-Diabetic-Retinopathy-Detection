import os
import json
import pandas as pd

def main():
    results_dir = "outputs/checkpoints"
    results = []
    
    # Common model names from the roadmap
    expected_models = ["resnet50", "vit_base_patch16_224", "mobilenetv3_small_100"]
    
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} not found. Run training first.")
        return
        
    for fname in os.listdir(results_dir):
        if fname.endswith("_results.json"):
            filepath = os.path.join(results_dir, fname)
            with open(filepath, 'r') as f:
                data = json.load(f)
                results.append(data)
                
    if not results:
        print(f"No *_results.json files found in {results_dir}.")
        return

    print("=== EarlyDR: Review 2 Model Comparison ===")
    
    # Formatting metrics for display
    summary = []
    for r in results:
        model_name = r.get("model", "Unknown")
        acc = r.get("test_accuracy", 0.0)
        qwk = r.get("test_qwk", 0.0)
        recall = r.get("test_recall", {})
        
        mild_recall = recall.get("Mild", 0.0)
        mod_recall = recall.get("Moderate", 0.0)
        
        summary.append({
            "Model": model_name,
            "Accuracy": f"{acc:.4f}",
            "QWK": f"{qwk:.4f}",
            "Mild Recall": f"{mild_recall:.4f}",
            "Moderate Recall": f"{mod_recall:.4f}"
        })
        
    df = pd.DataFrame(summary)
    try:
        print("\n" + df.to_markdown(index=False))
    except ImportError:
        print("\n" + df.to_string(index=False))
    
    print("\nNote: Early-stage sensitivity (Mild/Moderate Recall) is our primary clinical target.")

if __name__ == "__main__":
    main()
