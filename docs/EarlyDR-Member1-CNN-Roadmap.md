# EarlyDR — Member 1 Roadmap: CNN Track (ResNet / EfficientNet)

**Assigned architecture family:** Classical CNN backbone (ResNet-50 primary, EfficientNet-B0/B3 as comparison variant)
**Role in the 3-way comparison:** Represents the "high-capacity, well-established" baseline against which the ViT (Member 2) and MobileNet-edge (Member 3) tracks are measured.

---

## Objective
Train and tune a CNN-based classifier for 5-class DR severity grading, with the primary success criterion being **recall on Mild + Moderate DR**, not just overall accuracy — and produce a fully profiled model (architecture diagram, training curves, confusion matrix, per-class metrics) for Review 2.

## Milestones

### Milestone 1 — Setup & Baseline (Weeks 3–4)
- Set up training environment (PyTorch/TensorFlow, GPU access — Colab/Kaggle GPU or local).
- Load the team's frozen preprocessed dataset/splits (from Review 1 pipeline).
- Train a ResNet-50 baseline with standard cross-entropy loss and ImageNet-pretrained weights (transfer learning), no class balancing yet.
- Record baseline overall accuracy, per-class recall, and confusion matrix. This establishes the "naive" reference point that later steps must beat, especially on Mild/Moderate recall.

### Milestone 2 — Early-Stage-Focused Tuning (Weeks 4–5)
- Introduce class-weighted cross-entropy or focal loss to counter class imbalance and push Mild/Moderate recall up.
- Experiment with data augmentation targeted at minority classes (rotation, flips, brightness/contrast jitter, slight zoom — avoid distorting lesion patterns).
- Try EfficientNet-B0/B3 as an alternative backbone; compare against ResNet-50 on the same metric set.
- Track experiments (loss curves, per-class recall table) after each change so the improvement from each intervention is traceable.

### Milestone 3 — Model Selection & Deep Evaluation (Week 6)
- Select best-performing CNN configuration (ResNet vs EfficientNet, with/without class balancing) based on Mild+Moderate recall as primary criterion and overall accuracy/quadratic-weighted-kappa as secondary.
- Generate full evaluation artifacts: confusion matrix, per-class precision/recall/F1 table, ROC/PR curves, quadratic weighted kappa score.
- Run the model on a held-out cross-dataset test (e.g., trained on APTOS+EyePACS, tested on IDRiD) to report generalization behavior (feeds RQ3).

### Milestone 4 — Explainability & Profiling (Week 6–7)
- Generate Grad-CAM visualizations for a sample of correctly and incorrectly classified Mild/Moderate cases — this is both a novelty point and a debugging tool (are the "hot regions" anatomically plausible — microaneurysms/hemorrhages — or is the model cheating on artifacts?).
- Profile the model: parameter count, model file size, average inference time per image (CPU and GPU), to feed the final 3-way cost/accuracy comparison table.
- Draw/prepare the CNN architecture diagram for Review 2 slides.

### Milestone 5 — Review 2 Deliverables (Week 7–8)
- Model architecture diagram.
- Training/validation loss and accuracy curves (graphs).
- Confusion matrix + per-class metrics table.
- Grad-CAM sample visualizations.
- Written justification: why ResNet/EfficientNet was chosen, what tuning was done specifically for early-stage sensitivity, and how it compares to Member 2 (ViT) and Member 3 (MobileNet) on the shared metric set.

## Personal Risk Watch-Outs
- Don't stop at "overall accuracy looks good" — a high-accuracy model that only nails "No DR" and "Proliferative" while missing Mild/Moderate is a **failure for this project's specific goal**; always report the per-class breakdown.
- Keep every experiment's config (loss function, augmentation, backbone) logged — Review 2 requires justifying *why* the final choice was made, not just presenting the final number.
