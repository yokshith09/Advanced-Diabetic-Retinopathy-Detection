"""
Generates a comprehensive, publication-quality overall system architecture diagram
for the EarlyDR project. Covers:
  1. Input / Preprocessing pipeline
  2. Three parallel model tracks (ResNet50, ViT, MobileNetV3)
  3. Post-processing / Explainability layer
  4. Quantization / Edge deployment
  5. Output / Clinical Decision Support UI
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── colour palette ──────────────────────────────────────────────────────────
BG        = "#0D1117"
PANEL_BG  = "#161B22"
BORDER    = "#30363D"

C_INPUT   = "#1F6FEB"   # blue  – input / pre-processing
C_CNN     = "#E74C3C"   # red   – ResNet50
C_VIT     = "#3498DB"   # mid-blue – ViT
C_MOB     = "#2ECC71"   # green – MobileNet
C_POST    = "#9B59B6"   # purple – post-processing / explainability
C_OUTPUT  = "#F39C12"   # amber – output / UI
C_ARROW   = "#8B949E"   # grey arrows
C_TEXT    = "#E6EDF3"
C_SUB     = "#8B949E"


def fbox(ax, x, y, w, h, color, alpha=0.18, lw=1.8, radius=0.018):
    """Draw a rounded rectangle."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=color, alpha=alpha,
        edgecolor=color, linewidth=lw, zorder=3
    )
    ax.add_patch(box)
    return box


def label(ax, x, y, text, size=9, color=C_TEXT, bold=False, sub=False, ha="center", va="center"):
    weight = "bold" if bold else "normal"
    col    = C_SUB if sub else color
    ax.text(x, y, text, fontsize=size, color=col, fontweight=weight,
            ha=ha, va=va, zorder=5, wrap=True,
            fontfamily="DejaVu Sans")


def arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.5, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=4)


def curved_arrow(ax, x0, y0, x1, y1, color=C_ARROW, rad=0.25, lw=1.4):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=4)


# ── canvas ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 14))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.axis("off")

# ── TITLE ────────────────────────────────────────────────────────────────────
ax.text(11, 13.5, "EarlyDR — Overall System Architecture",
        fontsize=18, color=C_TEXT, fontweight="bold",
        ha="center", va="center", zorder=5,
        fontfamily="DejaVu Sans")
ax.text(11, 13.1,
        "End-to-end pipeline: Fundus Image → Preprocessing → Dual-Track DL Models → Explainability → Clinical Output",
        fontsize=9.5, color=C_SUB, ha="center", va="center", zorder=5,
        fontfamily="DejaVu Sans")

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 1 – INPUT / DATA
# ══════════════════════════════════════════════════════════════════════════════
COL1 = 1.5
fbox(ax, COL1, 10.5, 2.4, 1.5, C_INPUT, alpha=0.25)
label(ax, COL1, 11.1, "[INPUT]  Fundus Camera", size=9, bold=True, color=C_INPUT)
label(ax, COL1, 10.7, "Raw retinal fundus image", size=7.5, sub=True)
label(ax, COL1, 10.35, "JPEG / PNG  |  Varies: 224-4288 px", size=7, sub=True)

fbox(ax, COL1, 8.2, 2.4, 2.4, C_INPUT, alpha=0.2)
label(ax, COL1, 9.2,  "[DATA]  Datasets", size=9, bold=True, color=C_INPUT)
label(ax, COL1, 8.85, "• APTOS 2019  (3,662 imgs)", size=7.5, sub=True)
label(ax, COL1, 8.55, "• IDRiD  (516 imgs, IEEE)", size=7.5, sub=True)
label(ax, COL1, 8.25, "• Classes: No DR / Mild / Moderate", size=7.5, sub=True)
label(ax, COL1, 7.95, "         Severe / Proliferative DR", size=7.5, sub=True)

arrow(ax, COL1, 9.8,  COL1, 11.1 - 0.75, color=C_INPUT)   # Dataset → Fundus cam label
arrow(ax, COL1, 9.8,  COL1, 8.2  + 1.2,  color=C_INPUT)   # Down to dataset box (implicit)

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 2 – PRE-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
COL2 = 4.5

fbox(ax, COL2, 10.5, 2.6, 5.5, C_INPUT, alpha=0.18, lw=1.5)
label(ax, COL2, 13.0, "[PREPROC]  Preprocessing Pipeline", size=10, bold=True, color=C_INPUT)

steps = [
    ("1. Circular Crop & Centre",       "Detect fundus boundary, remove black borders"),
    ("2. Resize  →  224 × 224 px",      "Standardise to backbone input resolution"),
    ("3. Gaussian Blur  (σ=10)",        "Subtract local mean; enhance vessel contrast"),
    ("4. Colour Normalisation",         "ImageNet mean [.485, .456, .406] / std [.229, .224, .225]"),
    ("5. Data Augmentation (train)",    "H/V flip, ±20° rotate, brightness/contrast jitter"),
]
for i, (name, detail) in enumerate(steps):
    yy = 12.35 - i * 0.52
    ax.plot([3.3, 5.7], [yy + 0.05, yy + 0.05], color=BORDER, lw=0.7, zorder=3)
    label(ax, 4.5, yy - 0.05, f"  {name}", size=7.8, bold=True,
          color=C_INPUT, ha="center")
    label(ax, 4.5, yy - 0.3,  f"  {detail}", size=7,  sub=True, ha="center")

arrow(ax, 2.7, 10.5, 3.2, 10.5, color=C_INPUT)   # Input → Preproc

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 3 – MODEL TRACKS (three parallel)
# ══════════════════════════════════════════════════════════════════════════════
COL3 = 8.8
Y_CNN = 11.8
Y_VIT = 9.0
Y_MOB = 6.2

arrow(ax, 5.8, 10.5, 7.2, Y_CNN - 0.5, color=C_CNN)
arrow(ax, 5.8, 10.5, 7.2, Y_VIT,       color=C_VIT)
arrow(ax, 5.8, 10.5, 7.2, Y_MOB + 0.5, color=C_MOB)

# — ResNet50 ——————————————————————————
fbox(ax, COL3, Y_CNN, 3.0, 2.7, C_CNN, alpha=0.22)
label(ax, COL3, Y_CNN + 1.0, "[CNN]  CNN Baseline", size=10, bold=True, color=C_CNN)
label(ax, COL3, Y_CNN + 0.65, "ResNet50  (ImageNet pretrained)", size=8.5, sub=True)
rows_cnn = [
    ("Backbone:",  "ResNet-50  (25 M params, 90 MB)"),
    ("Loss:",      "Class-weighted Cross-Entropy"),
    ("LR Sched.:", "Cosine Annealing  T_max=15"),
    ("Novelty:",   "Grad-CAM explainability hook"),
    ("Test QWK:",  "0.7546   |   Mod. Recall 40.0%"),
]
for i, (k, v) in enumerate(rows_cnn):
    yy = Y_CNN + 0.2 - i * 0.38
    label(ax, COL3, yy, f"{k}  {v}", size=7.5, sub=(i > 0), ha="center")

# — ViT ———————————————————————————————
fbox(ax, COL3, Y_VIT, 3.0, 2.7, C_VIT, alpha=0.22)
label(ax, COL3, Y_VIT + 1.0, "[VIT]  Transformer Track", size=10, bold=True, color=C_VIT)
label(ax, COL3, Y_VIT + 0.65, "ViT-Base / Patch-16  (86 M params, 327 MB)", size=8, sub=True)
rows_vit = [
    ("Backbone:",  "12× Transformer Encoder blocks"),
    ("Attention:", "MHSA on 14×14 patch grid"),
    ("Loss:",      "Class-weighted Cross-Entropy"),
    ("Novelty:",   "Global attention → domain robust"),
    ("Test QWK:",  "0.8280   |   Mod. Recall 78.7%"),
]
for i, (k, v) in enumerate(rows_vit):
    yy = Y_VIT + 0.2 - i * 0.38
    label(ax, COL3, yy, f"{k}  {v}", size=7.5, sub=(i > 0), ha="center")

# — MobileNetV3 ———————————————————————
fbox(ax, COL3, Y_MOB, 3.0, 2.7, C_MOB, alpha=0.22)
label(ax, COL3, Y_MOB + 1.0, "[EDGE]  Edge Deployment Track", size=10, bold=True, color=C_MOB)
label(ax, COL3, Y_MOB + 0.65, "MobileNetV3-Small  (1.5 M params, 5.9 MB)", size=8, sub=True)
rows_mob = [
    ("Backbone:",   "Inv. Residuals + Squeeze-Excite"),
    ("Quant.:",     "FP16 → 3.01 MB  (−49.3%)"),
    ("CPU Latency:","10.63 ms  (FP32) / 262 ms (FP16)"),
    ("Novelty:",    "Edge-ready offline clinic deploy"),
    ("Test QWK:",   "0.8176   |   Mod. Recall 60.7%"),
]
for i, (k, v) in enumerate(rows_mob):
    yy = Y_MOB + 0.2 - i * 0.38
    label(ax, COL3, yy, f"{k}  {v}", size=7.5, sub=(i > 0), ha="center")

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 4 – POST-PROCESSING / EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
COL4 = 13.5

arrow(ax, 10.3, Y_CNN, 12.2, 10.5, color=C_POST)
arrow(ax, 10.3, Y_VIT, 12.2, 10.5, color=C_POST)
arrow(ax, 10.3, Y_MOB, 12.2, 10.5, color=C_POST)

fbox(ax, COL4, 10.5, 2.8, 6.0, C_POST, alpha=0.22, lw=1.5)
label(ax, COL4, 13.2, "[POST]  Post-Processing & Explainability", size=10, bold=True, color=C_POST)

postrows = [
    ("Grad-CAM Heatmap",     "Highlights retinal lesion regions\n(microaneurysms, haemorrhages) for ResNet50"),
    ("Attention-Map (ViT)",  "Self-attention rollout across 196 patches;\nshows globally attending regions"),
    ("Quantization (MobileNet)", "torch.half() FP16 compression;\n49.3% size reduction, clinically viable latency"),
    ("Cross-Dataset Test",   "Train on APTOS → blind eval on IDRiD;\nmeasures domain generalisation (RQ3)"),
]
for i, (name, detail) in enumerate(postrows):
    yy = 12.55 - i * 0.77
    fbox(ax, COL4, yy - 0.0, 2.6, 0.6, C_POST, alpha=0.12, lw=1.0, radius=0.01)
    label(ax, COL4, yy + 0.11, name,   size=8.0, bold=True, color=C_POST)
    label(ax, COL4, yy - 0.22, detail, size=7.2, sub=True)

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 5 – EVALUATION & METRICS
# ══════════════════════════════════════════════════════════════════════════════
COL5 = 17.5

arrow(ax, 14.9, 10.5, 16.1, 10.5, color=C_OUTPUT)

fbox(ax, COL5, 10.5, 3.2, 5.8, C_OUTPUT, alpha=0.20, lw=1.5)
label(ax, COL5, 13.2, "[EVAL]  Evaluation & Benchmarking", size=10, bold=True, color=C_OUTPUT)

label(ax, COL5, 12.75, "Primary Metric:  Quadratic Weighted Kappa (QWK)", size=8.3, bold=True, color=C_OUTPUT)
label(ax, COL5, 12.45, "Secondary:  Per-class Recall (Mild & Moderate DR)", size=7.8, sub=True)

# mini table
headers = ["Model",        "Acc",   "QWK",   "Mod.Rec"]
rows    = [
    ["ResNet50",           "0.682", "0.755", "40.0 %"],
    ["MobileNetV3 (FP16)", "0.751", "0.818", "60.7 %"],
    ["ViT-Base ★",         "0.791", "0.828", "78.7 %"],
]
col_x = [15.9, 17.0, 17.9, 18.9]
row_y0 = 11.9
ax.plot([15.5, 19.5], [row_y0 + 0.38, row_y0 + 0.38], color=C_OUTPUT, alpha=0.5, lw=0.9)
for ci, h in enumerate(headers):
    label(ax, col_x[ci], row_y0 + 0.18, h, size=7.5, bold=True, color=C_OUTPUT)
ax.plot([15.5, 19.5], [row_y0 + 0.0, row_y0 + 0.0], color=BORDER, lw=0.7)
for ri, row in enumerate(rows):
    yy = row_y0 - 0.45*(ri+1) + 0.3
    clr = C_VIT if "ViT" in row[0] else C_TEXT
    for ci, val in enumerate(row):
        label(ax, col_x[ci], yy, val, size=7.5, color=clr, bold=("ViT" in row[0]))
    if ri < 2:
        ax.plot([15.5, 19.5], [yy - 0.22, yy - 0.22], color=BORDER, lw=0.5, alpha=0.5)

label(ax, COL5, 10.1, "Cross-Dataset (IDRiD — domain shift):", size=8.0, bold=True, color=C_OUTPUT)
label(ax, COL5,  9.75, "ResNet50 QWK: 0.228  |  ViT QWK: 0.602  (best)", size=7.5, sub=True)
label(ax, COL5,  9.5,  "ViT retains 71.9% Moderate Recall on unseen camera data", size=7.3, sub=True)

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 6 – CLINICAL OUTPUT / UI
# ══════════════════════════════════════════════════════════════════════════════
COL6 = 17.5
Y_UI  = 7.2

arrow(ax, COL5, 9.3, COL6, Y_UI + 2.3, color=C_OUTPUT)

fbox(ax, COL6, Y_UI, 3.2, 4.2, C_OUTPUT, alpha=0.20, lw=1.5)
label(ax, COL6, Y_UI + 1.8,  "[OUT]  Clinical Output", size=10, bold=True, color=C_OUTPUT)
label(ax, COL6, Y_UI + 1.45, "Decision-Support UI (Streamlit/Gradio)", size=8, sub=True)
ui_items = [
    "• DR Severity Grade (0–4)",
    "• Per-class confidence scores",
    "• Referral recommendation",
    "  (Refer / Routine follow-up)",
    "• Grad-CAM heatmap overlay",
    "  (lesion localisation)",
    "• Inference latency display",
    "• Batch / Camp screening mode",
]
for i, item in enumerate(ui_items):
    label(ax, COL6, Y_UI + 1.0 - i*0.35, item, size=7.5, sub=(i % 2 == 1), ha="center")

# ══════════════════════════════════════════════════════════════════════════════
# BOTTOM – EDGE DEPLOYMENT NOTE
# ══════════════════════════════════════════════════════════════════════════════
fbox(ax, 8.0, 2.5, 7.0, 1.8, C_MOB, alpha=0.14, lw=1.2)
label(ax, 8.0, 3.25, "[EDGE DEPLOY]  Edge / Offline Deployment Path  (MobileNetV3 FP16)", size=9, bold=True, color=C_MOB)
label(ax, 8.0, 2.9,  "3.01 MB model  →  copies to any standard laptop or Android device over USB / SD-card", size=8, sub=True)
label(ax, 8.0, 2.6,  "No internet, no GPU, no cloud — fully offline inference for rural primary health centres", size=7.8, sub=True)
label(ax, 8.0, 2.3,  "CPU latency: ~262 ms/image (FP16)  •  GPU latency: 22 ms/image  •  acceptable for clinic workflow", size=7.5, sub=True)

arrow(ax, COL3, Y_MOB - 1.35, 8.0, 3.38, color=C_MOB)

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (C_INPUT, "Input / Preprocessing"),
    (C_CNN,   "ResNet50 CNN Track"),
    (C_VIT,   "ViT Transformer Track"),
    (C_MOB,   "MobileNetV3 Edge Track"),
    (C_POST,  "Explainability / Post-proc"),
    (C_OUTPUT,"Evaluation & Clinical UI"),
]
lx, ly = 0.3, 5.8
for i, (col, lbl) in enumerate(legend_items):
    yy = ly - i * 0.48
    ax.add_patch(FancyBboxPatch((lx, yy - 0.13), 0.28, 0.26,
                                boxstyle="round,pad=0.01",
                                facecolor=col, alpha=0.6, edgecolor=col,
                                linewidth=1, zorder=5))
    label(ax, lx + 0.18, yy + 0.0, f"  {lbl}", size=7.8, ha="left", sub=True)

# ── watermark / credit ───────────────────────────────────────────────────────
ax.text(21.7, 0.2, "EarlyDR Project  |  Review 2  |  2026",
        fontsize=7, color="#484F58", ha="right", va="bottom", zorder=5)

plt.tight_layout(pad=0.3)
out = "outputs/overall_architecture.png"
plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved -> {out}")
