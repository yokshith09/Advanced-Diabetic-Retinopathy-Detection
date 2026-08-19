"""
Combines all three members' results.json files (from train.py) into one
comparison table + bar chart for the Review 2 report.

Usage (run once everyone has trained):
    python scripts/compare_results.py \
        --results outputs/checkpoints/resnet50_results.json \
                   outputs/checkpoints/vit_base_patch16_224_results.json \
                   outputs/checkpoints/mobilenetv3_small_100_results.json
"""

import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# Reference points from published literature (APTOS), for context in the report
BENCHMARKS = {
    "ResNet50 (paper)": 0.901,
    "DenseNet121 (paper)": 0.908,
    "EfficientNetB0, tuned (paper)": 0.921,
    "MobileNetV3+ordinal head (paper)": 0.90,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True,
                         help="Paths to each member's *_results.json")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for path in args.results:
        with open(path) as f:
            r = json.load(f)
        row = {
            "model": r["model"],
            "test_accuracy": r["test_accuracy"],
            "test_qwk": r["test_qwk"],
        }
        for cls in CLASS_NAMES:
            row[f"recall_{cls}"] = r["test_recall"].get(cls, None)
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n=== Model comparison ===")
    print(df.to_string(index=False))

    csv_path = os.path.join(args.out_dir, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved table to {csv_path}")

    # --- QWK bar chart, with your models plus the literature benchmarks ---
    fig, ax = plt.subplots(figsize=(9, 5))
    your_models = dict(zip(df["model"], df["test_qwk"]))
    all_labels = list(your_models.keys()) + list(BENCHMARKS.keys())
    all_values = list(your_models.values()) + list(BENCHMARKS.values())
    colors = ["#C4661F"] * len(your_models) + ["#5C736B"] * len(BENCHMARKS)
    bars = ax.barh(all_labels, all_values, color=colors)
    ax.set_xlabel("Quadratic Weighted Kappa (QWK)")
    ax.set_title("EarlyDR models vs. published benchmarks (APTOS)")
    ax.set_xlim(0, 1.0)
    for b, v in zip(bars, all_values):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    chart_path = os.path.join(args.out_dir, "qwk_comparison_chart.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved comparison chart to {chart_path}")

    # --- Mild/Moderate recall focus table, the project's core novelty metric ---
    focus_df = df[["model", f"recall_Mild", f"recall_Moderate"]].copy()
    focus_df["early_stage_avg_recall"] = focus_df[["recall_Mild", "recall_Moderate"]].mean(axis=1)
    print("\n=== Early-stage (Mild/Moderate) recall — the project's core claim ===")
    print(focus_df.to_string(index=False))


if __name__ == "__main__":
    main()
