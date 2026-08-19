# EarlyDR — Member 2 Roadmap: Vision Transformer (ViT) Track

**Assigned architecture family:** Vision Transformer (ViT-Base/16, pretrained; DeiT or Swin as optional comparison variant)
**Role in the 3-way comparison:** Represents the "modern, attention-based, higher accuracy potential but data/compute hungry" option — tests whether global attention over the fundus image helps catch subtle early-stage lesions better than a CNN's local receptive fields.

---

## Objective
Fine-tune a Vision Transformer for 5-class DR severity grading on the shared dataset, evaluate it against the same early-stage-recall-first criterion as the CNN and MobileNet tracks, and investigate whether ViT's attention maps offer better/different interpretability for Mild/Moderate cases than CNN Grad-CAM.

## Milestones

### Milestone 1 — Setup & Baseline (Weeks 3–4)
- Set up a ViT fine-tuning environment (Hugging Face `transformers`/`timm`, GPU access).
- Load the frozen shared preprocessed dataset/splits.
- Fine-tune a pretrained ViT-Base/16 (ImageNet-21k pretrained) on the 5-class task with standard cross-entropy — this baseline is directly comparable to Member 1's ResNet baseline (same data, same starting loss function).
- Note: ViTs typically need more data/augmentation than CNNs to avoid overfitting on a comparatively small dataset like APTOS alone — plan to train on the combined APTOS+EyePACS pool if the baseline underperforms due to data scarcity.

### Milestone 2 — Early-Stage-Focused Tuning (Weeks 4–5)
- Apply the same class-weighted loss / focal loss strategy as the CNN track for a fair comparison (isolate architecture effect from loss-function effect).
- Tune ViT-specific hyperparameters: patch size, learning rate warm-up/schedule (ViTs are sensitive to LR schedule), and stronger regularization (dropout, stochastic depth) to combat overfitting on the smaller dataset.
- Optionally compare a hybrid/hierarchical variant (Swin Transformer) if plain ViT overfits or underperforms — Swin's local-window attention may suit fundus images better.

### Milestone 3 — Model Selection & Deep Evaluation (Week 6)
- Select the best ViT configuration using Mild+Moderate recall as the primary criterion (same metric protocol as Member 1, for apples-to-apples comparison).
- Generate confusion matrix, per-class precision/recall/F1, ROC/PR curves, and quadratic weighted kappa.
- Run the same cross-dataset generalization test (train on APTOS+EyePACS, test on IDRiD) to compare ViT's domain-shift robustness against the CNN's (feeds RQ3).

### Milestone 4 — Explainability & Profiling (Week 6–7)
- Generate attention-rollout / attention-map visualizations for sample Mild/Moderate cases (the ViT equivalent of Grad-CAM) — compare qualitatively against Member 1's Grad-CAM outputs: does the Transformer attend to different retinal regions?
- Profile the model: parameter count, model size, inference time per image (CPU and GPU) — ViT is expected to be heavier than the CNN and much heavier than MobileNet; this is an expected and useful data point for the final trade-off table, not a failure.
- Prepare the ViT architecture diagram (patch embedding → transformer encoder stack → classification head) for Review 2.

### Milestone 5 — Review 2 Deliverables (Week 7–8)
- Model architecture diagram.
- Training/validation curves (graphs).
- Confusion matrix + per-class metrics table.
- Attention-map visualizations on sample cases.
- Written justification: why ViT was chosen as the comparison point, what was needed to make it competitive on a comparatively small medical dataset, and how its accuracy/cost trade-off compares to Member 1 (CNN) and Member 3 (MobileNet).

## Personal Risk Watch-Outs
- ViTs are known to underperform CNNs on small datasets without heavy augmentation or pretraining — don't be surprised if the raw baseline is *worse* than Member 1's; the tuning work (Milestone 2) is where the real comparison story comes from.
- Track and report inference latency/model size honestly even if it's the "expensive" option in the 3-way comparison — that trade-off is itself a key project finding, not something to downplay.
