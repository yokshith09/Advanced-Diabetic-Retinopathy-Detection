# EarlyDR — Member 3 Roadmap: Lightweight MobileNet (Edge Deployment) Track

**Assigned architecture family:** MobileNetV2/V3 (small/large variant), with post-training quantization (INT8/TFLite) for edge deployment
**Role in the 3-way comparison:** Represents the "deployability-first" option — directly answers the project's cost-vs-accuracy and edge-hardware feasibility angle (RQ2/RQ4), the piece that differentiates EarlyDR from most academic DR papers.

---

## Objective
Train a lightweight MobileNet classifier for 5-class DR severity grading, quantize/optimize it for edge inference, and rigorously measure the accuracy trade-off against Member 1 (CNN) and Member 2 (ViT) so the team can present a genuine cost-vs-accuracy curve, not just three separate accuracy numbers.

## Milestones

### Milestone 1 — Setup & Baseline (Weeks 3–4)
- Set up training environment (TensorFlow/Keras or PyTorch with TFLite/ONNX export path — decide early since this affects the quantization toolchain later).
- Load the frozen shared preprocessed dataset/splits.
- Fine-tune MobileNetV2 (ImageNet-pretrained) on the 5-class task with standard cross-entropy as the baseline, directly comparable to Member 1's and Member 2's baselines.

### Milestone 2 — Early-Stage-Focused Tuning (Weeks 4–5)
- Apply the same class-weighted/focal loss strategy used by the other two tracks for a fair comparison.
- Compare MobileNetV2 vs MobileNetV3-Small vs MobileNetV3-Large to find the best accuracy/size trade-off point *before* quantization.
- Keep an explicit log of accuracy vs. parameter count at this stage — this pre-quantization comparison is itself useful for the final report.

### Milestone 3 — Model Selection & Deep Evaluation (Week 6)
- Select the best full-precision MobileNet configuration using Mild+Moderate recall as the primary criterion (same protocol as the other two tracks).
- Generate confusion matrix, per-class precision/recall/F1, ROC/PR curves, and quadratic weighted kappa.
- Run the same cross-dataset generalization test (train on APTOS+EyePACS, test on IDRiD) for the domain-shift comparison (RQ3).

### Milestone 4 — Quantization & Edge Profiling (Week 6–7) — *this track's unique deliverable*
- Apply post-training quantization (INT8) and/or quantization-aware training to the selected MobileNet model; export to TFLite (or ONNX + runtime equivalent).
- Re-evaluate the quantized model on the same test set and **explicitly report the accuracy/recall drop caused by quantization** — this is the core novelty artifact for the whole project (RQ4).
- Benchmark inference latency and model file size on the lowest-spec hardware available to the team (e.g., a laptop CPU-only run, or Raspberry Pi if accessible) to simulate a clinic-hardware constraint.
- Generate Grad-CAM (MobileNet is CNN-based, so Grad-CAM applies directly) on sample Mild/Moderate cases, both pre- and post-quantization, to check whether quantization degrades the model's attention to lesion regions.

### Milestone 5 — Review 2 Deliverables (Week 7–8)
- Model architecture diagram (full-precision and quantized pipeline).
- Training/validation curves (graphs), plus a pre- vs post-quantization accuracy/recall comparison table.
- Confusion matrix + per-class metrics table (both full-precision and quantized).
- Model size and inference latency benchmarks (the key numbers for the team's final 3-way trade-off chart).
- Written justification: why MobileNet + quantization was chosen to represent the "deployability" arm, what accuracy cost quantization imposed, and how this compares against Member 1 (CNN) and Member 2 (ViT) on the shared metric set — this comparison table is the team's central Review 2 output.

## Personal Risk Watch-Outs
- Quantization can silently hurt the *minority classes* (Mild/Moderate) more than overall accuracy suggests — always re-check the per-class recall after quantization, not just the aggregate number.
- Coordinate with Members 1 & 2 early on the exact test set and metric definitions — this track's whole value proposition is the head-to-head comparison table, so metric mismatches between tracks would undermine the entire project's headline result.
