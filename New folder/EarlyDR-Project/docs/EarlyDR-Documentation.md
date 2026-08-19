# EarlyDR — Diabetic Retinopathy Early-Stage Detection

**Slogan:** *"Catch it before it steals the sight."*

---

## 1. Introduction

Diabetic Retinopathy (DR) is a progressive complication of diabetes mellitus that damages the blood vessels of the retina. It is one of the leading causes of preventable blindness worldwide, particularly in low- and middle-income countries where regular ophthalmic screening is inaccessible. EarlyDR is a deep-learning-based screening assistant that analyzes retinal fundus images and classifies them into standard DR severity grades — **No DR, Mild, Moderate, Severe, Proliferative DR** — with a deliberate design emphasis on correctly catching the **Mild and Moderate (early)** stages, where intervention can still prevent vision loss.

The system is intended as a **decision-support tool for low-resource clinics and primary health centers**, not a replacement for an ophthalmologist — flagging at-risk patients for referral before symptoms become irreversible.

---

## 2. Problem Statement

Diabetic retinopathy is largely asymptomatic in its early stages. Patients typically do not notice vision changes until the disease has progressed to Moderate/Severe stages, by which point damage may already be partially irreversible. Screening requires a trained ophthalmologist and fundus camera — both scarce outside urban centers. Most existing automated DR-screening research and products are optimized for overall classification accuracy, which is dominated by the "No DR" and "Severe/Proliferative" classes (easiest to distinguish), while **Mild and Moderate cases — the most clinically actionable ones — are the most frequently misclassified**. EarlyDR directly targets this gap: building a model whose primary evaluation criterion is early-stage sensitivity/recall, not aggregate accuracy, while remaining light enough to run on modest clinic hardware.

---

## 3. Motivation

- Blindness from DR is almost entirely preventable **if caught early** — a screening delay of even 1–2 years can be the difference between a manageable condition and permanent vision loss.
- India, along with much of South Asia and Sub-Saharan Africa, carries a disproportionately high diabetic population with a severe shortage of retina specialists relative to patient load.
- Smartphone-based and low-cost fundus cameras are becoming increasingly available in primary health centers, but the bottleneck is **interpretation**, not image capture.
- Publicly available, well-annotated fundus datasets (APTOS, EyePACS, IDRiD) make it feasible for a student team to build and rigorously evaluate a clinically meaningful model without needing to collect new patient data.
- This project maps directly onto UN SDG 3 (Good Health and Well-being) while also giving the team hands-on experience comparing CNN, Transformer, and edge-optimized architectures on a single, medically meaningful task.

---

## 4. Research Questions / Gaps to be Addressed

1. **RQ1 — Early-stage sensitivity:** Can a model be trained/tuned to significantly improve recall on Mild and Moderate DR classes without collapsing overall performance, compared to models trained purely for accuracy?
2. **RQ2 — Architecture trade-off:** How do CNN-based (ResNet/EfficientNet), Transformer-based (ViT), and lightweight edge (MobileNet, quantized) architectures compare on accuracy, early-stage recall, inference latency, and model size?
3. **RQ3 — Cross-dataset generalization:** Does a model trained on APTOS generalize to EyePACS/IDRiD images, which differ in camera type, resolution, and population — a common real-world failure mode of DR-screening research?
4. **RQ4 — Deployability:** Can a quantized/pruned version of the best model run inference in near real-time on low-cost/edge hardware (e.g., a basic laptop or Raspberry-Pi-class device) typical of a rural clinic, with acceptable accuracy degradation?

**Gap addressed:** Most published DR-grading literature reports macro-accuracy or quadratic-weighted-kappa as the headline metric, which under-represents performance on the clinically critical early classes. Few papers explicitly optimize and report **per-class recall for Mild/Moderate DR alongside a deployability/cost analysis** — EarlyDR treats both as first-class evaluation criteria.

---

## 5. Related Products

| Product | URL | Features | Limitations |
|---|---|---|---|
| **IDx-DR (Digital Diagnostics)** | digitaldiagnostics.com | FDA-cleared autonomous AI system for DR detection from fundus images; used in primary-care settings | Requires proprietary fundus camera (Topcon NW400); expensive; binary "refer / no refer" output rather than fine-grained severity grading |
| **Google/Verily ARDA (Automated Retinal Disease Assessment)** | research.google/teams/health | Deep learning pipeline validated across Thailand/India screening programs; strong published accuracy on referable DR | Not a publicly purchasable product for small clinics; large model, not optimized for edge/offline use; validation tied to specific camera types |
| **EyeArt (Eyenuk)** | eyenuk.com | FDA-cleared, fully autonomous DR detection with fast turnaround; supports multiple camera types | Cloud-dependent workflow in most deployments; subscription/licensing cost is a barrier for low-resource clinics |
| **Medios AI (Remidio)** | remidio.com | Smartphone-based fundus imaging + AI grading, designed specifically for low-resource/rural screening in India | Tied to Remidio's own hardware ecosystem; severity grading granularity and offline/edge performance vary by deployment |

**Positioning of EarlyDR relative to these:** all four are either camera-locked, cloud-dependent, subscription-gated, or optimized for binary referral rather than graded early-stage sensitivity. EarlyDR is explicitly designed to be **hardware-agnostic (works on any reasonably standard fundus image), open, and tuned for early-stage recall with a lightweight edge-deployable variant** — a combination not offered by the products above.

---

## 6. SDG Mapping

### 6.1 SDG Identified
**SDG 3 — Good Health and Well-being** (specifically Target 3.4: reduce premature mortality/disability from non-communicable diseases through prevention and treatment, and Target 3.8: access to quality essential health-care services).

### 6.2 How the Case Study Relates to the SDG
Diabetic retinopathy is a direct complication of diabetes, a major non-communicable disease (NCD). By enabling early, low-cost, accessible screening, EarlyDR contributes to:
- **Reducing preventable blindness**, a disability outcome directly targeted under SDG 3.4.
- **Improving access to essential eye-care services** in low-resource settings lacking retina specialists (SDG 3.8, universal health coverage).
- Supporting **task-shifting** — enabling non-specialist health workers to perform a first-line screening pass, reserving scarce ophthalmologist time for confirmed/high-risk referrals.

---

## 7. Dataset Details

**Primary sources:**
- **APTOS 2019 Blindness Detection** (Kaggle / Asia Pacific Tele-Ophthalmology Society) — ~3,662 labeled training fundus images, 5-class severity grading (0–4).
- **EyePACS (Kaggle Diabetic Retinopathy Detection)** — ~35,000+ labeled fundus images from the EyePACS screening network, same 5-class grading scale; larger and more heterogeneous than APTOS.
- **IDRiD — Indian Diabetic Retinopathy Image Dataset** (hosted on **IEEE Dataport**) — smaller, high-quality, expert-annotated dataset including DR severity grading, DME (macular edema) grading, and pixel-level lesion segmentation (microaneurysms, hemorrhages, exudates). IEEE Dataport reference sample to be attached separately for Review 1.

### 7.1 Dataset Description
Across these three sources, the team has access to fundus (retinal) photographs captured under varying camera equipment, lighting, and population conditions, each labeled on the standard International Clinical DR Severity Scale (No DR / Mild / Moderate / Severe / Proliferative DR). APTOS offers a clean, moderate-sized, competition-quality dataset ideal for initial model development; EyePACS offers scale and real-world noise/heterogeneity useful for robustness testing; IDRiD offers the highest label quality along with lesion-level segmentation masks, useful both as a smaller high-trust validation set and, later, as an explainability/localization aid (highlighting *why* a case is graded Mild/Moderate). Using all three together enables both **within-dataset training/validation** and **cross-dataset generalization testing (RQ3)**.

### 7.2 Specific Challenges Addressed in the Dataset
- **Severe class imbalance:** "No DR" and "Moderate" dominate; "Severe" and "Proliferative" are comparatively rare across all three sources — requires class-weighted loss, oversampling of minority classes, or focal loss.
- **Image quality/noise:** variable illumination, blur, artifacts (dust, reflections), and off-center/cropped fundus captures, especially in EyePACS — requires quality-filtering and normalization (CLAHE contrast enhancement, circular fundus cropping, resizing).
- **Label subjectivity/noise:** inter-grader disagreement on borderline Mild vs. Moderate cases is a documented issue in EyePACS in particular — motivates using IDRiD (expert-graded, higher label trust) as a cleaner validation/test benchmark.
- **Camera/domain shift:** APTOS, EyePACS, and IDRiD differ in camera hardware and population, so a model trained purely on one will show accuracy drop on another — this is the basis for the cross-dataset generalization experiment (RQ3).
- **Preprocessing pipeline required:** standardization steps planned — circular crop to remove black borders, resize to model input resolution, CLAHE/green-channel enhancement to improve vessel/lesion contrast, and normalization to the backbone's expected input distribution.

---

## 8. Novelty in the Product

1. **Early-stage-first evaluation:** unlike most published DR classifiers that report overall accuracy/kappa, EarlyDR's primary success metric is **recall/sensitivity on Mild + Moderate DR specifically**, using class-weighted loss and a custom evaluation dashboard that reports per-class metrics rather than a single aggregate number.
2. **Three-way architecture comparison on one pipeline:** the team directly compares a classical CNN (ResNet/EfficientNet), a Vision Transformer, and a lightweight quantized MobileNet on the *same* preprocessed dataset and the *same* early-stage-recall metric — producing an explicit accuracy-vs-cost trade-off curve (directly serving the deployability requirement).
3. **Edge-deployability as a design constraint, not an afterthought:** the MobileNet variant is quantized (e.g., INT8 / TFLite) with the explicit goal of running inference on low-cost clinic hardware, addressing a practical deployment gap that most academic DR papers do not evaluate.
4. **Cross-dataset robustness testing:** training on one dataset and validating on another (e.g., train on APTOS, test on IDRiD) to explicitly measure and report generalization/domain-shift, which is rarely reported in student-level DR projects.

---

## 9. UI/UX Concept (for demo purposes)

A minimal web interface (planned for later reviews, e.g., Streamlit/Gradio prototype) for demonstrating the model to a clinic worker:

- **Upload screen:** drag-and-drop or camera-capture of a fundus image.
- **Result screen:** predicted severity grade (No DR / Mild / Moderate / Severe / Proliferative) with a confidence score per class, plus a plain-language referral recommendation ("Refer to ophthalmologist" / "Routine follow-up").
- **Explainability overlay:** a Grad-CAM heatmap over the fundus image highlighting the retinal regions that most influenced the model's prediction, so the clinic worker can visually sanity-check the flagged area (hemorrhage/exudate location).
- **Batch mode (stretch goal):** allow a folder of images to be screened at once for camp-style screening drives.

The UI is a demonstration layer only — the core deliverable for this course is the model pipeline and evaluation, not production software engineering.

---

## 10. Review-1 Roadmap (Team, Up to Dataset Acquisition)

| Phase | Task | Owner | Target |
|---|---|---|---|
| 1 | Finalize problem statement, SDG mapping, and related-product research (this document) | Whole team | Week 1 |
| 2 | Download and inspect APTOS 2019, EyePACS, and IDRiD (IEEE Dataport) datasets; verify label formats and class distributions | Whole team (split by dataset) | Week 1–2 |
| 3 | Exploratory Data Analysis (EDA): class distribution plots, sample image grids per class, image resolution/quality audit | Whole team | Week 2 |
| 4 | Build shared preprocessing pipeline: circular crop, resize, CLAHE enhancement, normalization, train/val/test split strategy (with class-weighting plan) | Whole team, one owner finalizes script | Week 2–3 |
| 5 | Freeze the shared preprocessed dataset + splits so all 3 members train on identical data for a fair model comparison | Whole team | Week 3 |
| 6 | Documentation finalization + IEEE Dataport sample attachment for submission | Whole team | Before Review 1 deadline |
| 7 | Individual model roadmaps (see per-member documents) begin in parallel from Week 3 onward, feeding into Review 2 | Each member | Ongoing → Review 2 |

**Critical dependency:** Phase 5 (frozen shared preprocessed dataset/splits) must be locked before any member starts serious training — this is what makes the Review 2 model comparison scientifically valid.

