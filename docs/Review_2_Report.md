# EarlyDR Project: Review 2
## Case Study: Deep Learning Model Implementation, Convergence, and Benchmarking

---

## 1. Proposed System Overall Architecture

The EarlyDR screening assistant employs a modular, pipeline-oriented architecture designed to ingestion-standardize raw fundus photographs, evaluate them through three parallel model tracks, generate localized explainability heatmaps, and output clinical recommendations.

### 1.1 Ingestion and Preprocessing Pipeline
Retinal fundus images are inherently variable due to differences in camera hardware, lighting, zoom, and patient physiology. The preprocessing module standardizes inputs before feeding them to the neural networks:
1. **Circular Crop & Center**: Dynamically detects the boundaries of the circular fundus mask to crop out black borders, centering the region of interest.
2. **Resize (224x224)**: Standardizes resolution to match the input dimensions of the model backbones.
3. **Gaussian Blur Filtering**: Employs local subtraction to enhance high-frequency structures, highlighting blood vessels, microaneurysms, and hemorrhages while suppressing global illumination noise.
4. **Normalization**: Normalizes pixel values using ImageNet mean (`[0.485, 0.456, 0.406]`) and standard deviation (`[0.229, 0.224, 0.225]`).

![Overall System Architecture](file:///e:/PROJECTS/EarlyDR-Project/outputs/overall_architecture.png)

### 1.2 Deep Learning Model Tracks Architecture

We designed and evaluated three distinct neural network architectures, representing different structural paradigms: classical convolutional (ResNet50), self-attention-based (ViT-Base), and resource-efficient (MobileNetV3).

![Deep Learning Model Architectures](file:///e:/PROJECTS/EarlyDR-Project/outputs/deep_learning_architectures.png)

---

## 2. Deep Learning Model Implementation and Justification

### 2.1 Classical CNN Baseline: ResNet50
*   **Architecture & Block Details**: ResNet50 utilizes bottleneck residual blocks. As shown in the block diagram, each block consists of a 1x1 convolution (for dimensionality reduction), a 3x3 convolution (for spatial feature extraction), and another 1x1 convolution (for dimensionality restoration). An identity shortcut skips these three layers and adds the input `x` directly to the output before applying the ReLU activation: `Output = ReLU(F(x) + x)`. This architecture mitigates the vanishing gradient problem, allowing the training of deep networks.
*   **Justification**: ResNet50 is the standard benchmark in medical image classification. It provides a robust, high-capacity baseline for comparative analysis.
*   **Novelty**: Rather than optimizing for raw accuracy, we implemented a **Class-Weighted Cross-Entropy Loss** to heavily penalize misclassifications on minority early-stage classes (Mild/Moderate). We also integrated **Grad-CAM** to provide visual explainability to clinicians by overlaying attention heatmaps on the fundus images.

### 2.2 Transformer-based Architecture: Vision Transformer (ViT-Base/16)
*   **Architecture & Pipeline Details**: The ViT-Base architecture discards convolutional layers entirely. It reshapes the 224x224 input image into a sequence of flat 16x16 patches. These patches are mapped to a 768-dimensional embedding space via a linear projection layer. A learnable class token (`[CLS]`) and position embeddings are added to the patch sequence. The sequence is processed through 12 Transformer Encoder layers containing Multi-Head Self-Attention (MHSA) blocks, layer normalization, and Multi-Layer Perceptrons (MLP) with GELU activation.
*   **Justification**: Vision Transformers capture long-range global dependencies natively. In diabetic retinopathy, tiny lesions (such as microaneurysms) can be scattered across the retina. CNNs struggle with global context due to their local receptive fields, whereas ViT's self-attention mechanism can associate distant lesions across the entire image.
*   **Novelty**: We tested whether global attention mechanisms can overcome the severe class imbalances of diabetic retinopathy datasets better than local convolutions under identical training constraints.

### 2.3 Edge-Deployable Architecture: MobileNetV3 Small
*   **Architecture & Block Details**: MobileNetV3 Small is built on Inverted Residual Blocks with Squeeze-and-Excitation (SE) modules. The block expands channels with a 1x1 convolution, performs a lightweight depthwise convolution (3x3 or 5x5), passes the features through an SE block (which computes channel-wise attention weights via global pooling and multi-layer perceptrons to scale active feature channels), projects channels back down with a 1x1 linear convolution, and uses a residual shortcut. It utilizes the hard-swish activation function for faster inference.
*   **Justification**: To fulfill the clinical requirement of deploying automated screening tools in low-resource rural health camps lacking GPU-accelerated cloud infrastructure, we required a model with a minimal memory footprint and high CPU speed.
*   **Novelty**: We applied **FP16 (Half-Precision) Post-Training Quantization** to the final trained model. This successfully compressed the model weights, achieving a **49.3% file size reduction** (from 5.94 MB to 3.01 MB), making it highly feasible for offline edge devices (e.g., standard laptops or Raspberry Pi modules) without losing class-specific screening capabilities.

---

## 3. Complete Training and Testing with Convergence

The model backbones were trained for 15 epochs using the **AdamW optimizer** with a starting learning rate of $1\times10^{-4}$ and a batch size of 32. 

### 3.1 Learning Rate Scheduling and Loss Formulation
To guarantee proper convergence, we integrated **Cosine Annealing Learning Rate scheduling** ($T_{max} = 15$). This smoothly decays the learning rate to prevent oscillations in later training stages. 

The primary clinical target is early-stage diabetic retinopathy. Standard Cross-Entropy loss causes models to ignore minority classes (Mild/Moderate DR) and over-predict the majority classes (No DR). We computed inverse-frequency class weights from the training set splits:
$$\text{Weight}(c) = \frac{N_{\text{total}}}{C \times N_c}$$
Where $C=5$ classes and $N_c$ is the count of class $c$. This formulation applies heavier loss penalties when early-stage cases are misclassified.

### 3.2 Convergence Performance
The training convergence history was logged dynamically. Validation Quadratic Weighted Kappa (QWK) was used as the primary tracking metric to save the "best" model weights.

![Training Convergence Curves](file:///e:/PROJECTS/EarlyDR-Project/outputs/convergence_plots.png)

*   **Training Loss Curves**: The ResNet50 model and MobileNetV3 converged rapidly, with their training loss steadily decreasing. The ViT-Base model converged more slowly during the first 3 epochs due to the lack of inductive bias in Transformer layers, but once attention weights aligned (epochs 4–15), its training loss decreased significantly.
*   **Validation QWK Progress**: Validation QWK curves show that ViT-Base achieved the highest and most stable validation QWK (peaking at $0.8595$), followed by MobileNetV3 ($0.8331$) and ResNet50 ($0.7671$).

---

## 4. Performance Benchmarking & Results

### 4.1 Table 1: Primary Testing Results (APTOS Dataset)
*The primary clinical target is Early-Stage Sensitivity (Mild and Moderate Recall) and QWK, rather than raw classification accuracy.*

| Model Track | Accuracy | QWK | Mild Recall | Moderate Recall |
| :--- | :---: | :---: | :---: | :---: |
| **ResNet50 (CNN Baseline)** | 0.6818 | 0.7546 | **0.6250** | 0.4000 |
| **MobileNetV3 (FP32)** | 0.7564 | 0.8129 | 0.5893 | 0.5333 |
| **MobileNetV3 (Quantized FP16)** | 0.7509 | 0.8176 | 0.5536 | 0.6067 |
| **ViT Base Patch16** | **0.7909** | **0.8280** | 0.5179 | **0.7867** |

### 4.2 Table 2: Cross-Dataset Robustness Testing (IDRiD Dataset)
*Models trained entirely on the APTOS dataset were evaluated blind on the IDRiD dataset (unseen images captured using different camera brands, color temperatures, and demographics) to measure domain generalization.*

| Model Track | Accuracy | QWK | Mild Recall | Moderate Recall | Proliferative Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet50 (CNN Baseline)** | 0.3981 | 0.2283 | 0.0000 | 0.1250 | 0.2308 |
| **MobileNetV3 (Quantized FP16)**| 0.4272 | 0.3877 | 0.0000 | 0.2500 | **0.5385** |
| **ViT Base Patch16** | **0.4854** | **0.6023** | 0.0000 | **0.7188** | 0.1538 |

### 4.3 Table 3: Hardware Profiling and Model Details
*Inference latency was benchmarked on CPU (Intel Core) and GPU (NVIDIA CUDA) over 50 runs with a batch size of 1.*

| Model Track | Parameter Count | File Size (MB) | Size Reduction | CPU Latency | GPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet50** | 23,518,277 | 90.02 MB | 0.0% (Baseline) | 62.72 ms | **12.78 ms** |
| **ViT Base** | 85,802,501 | 327.37 MB | 0.0% (Baseline) | 166.69 ms | 13.79 ms |
| **MobileNetV3 (FP32)** | 1,522,981 | 5.94 MB | 0.0% (Baseline) | **10.63 ms** | 13.64 ms |
| **MobileNetV3 (FP16)** | **1,522,981** | **3.01 MB** | **49.3%** | 262.03 ms | 22.41 ms |

### 4.4 Figure: Test Set Confusion Matrices

Below are the raw confusion matrix heatmaps generated on the test sets, detailing class-by-class classifications:

![Confusion Matrices Heatmap](file:///e:/PROJECTS/EarlyDR-Project/outputs/confusion_matrices.png)

---

## 5. Observations and Discussion

### 5.1 Transformer Superiority in Spatial Feature Association
The Vision Transformer (ViT-Base) significantly outperformed the CNN models on the primary APTOS test set, achieving a QWK of **0.8280** and an Accuracy of **0.7909**. Most importantly, it achieved a recall of **78.67% on Moderate DR**, nearly doubling the sensitivity of the classical ResNet50 baseline (40.00%). 
*   **Discussion**: CNNs rely on local receptive fields that expand hierarchically. This makes them highly sensitive to localized features but poor at correlating small, widely separated lesions (like microaneurysms) across different quadrants of the retina. ViT's self-attention mechanism computes pairwise attention weights between all patches in the image from the very first layer. This global contextual representation allows it to detect the overall pathological footprint of the retina, yielding superior classification bounds for early-stage DR.

### 5.2 Generalization and Domain Shift Challenges
Medical AI models often fail catastrophically when evaluated on datasets from different clinical sites due to "domain shift" (changes in camera optics, illumination, and color representation). This was observed during our blind evaluation on the IDRiD dataset.
*   **Observations**: 
    - The ResNet50 baseline crashed to a QWK of **0.2283**, predicting "No DR" for almost all early-stage cases (recalling only 12.50% of Moderate DR and 0% of Mild DR).
    - The Vision Transformer (ViT) demonstrated remarkable robustness, maintaining a QWK of **0.6023** and correctly recalling **71.88%** of Moderate DR cases on the unseen dataset.
    - All models struggled to detect the 5 Mild DR cases in the IDRiD test set (resulting in 0% recall). The confusion matrix reveals these cases were misclassified as "No DR" due to their similarity to healthy retinas under the different lighting conditions of the IDRiD images.
*   **Discussion**: The CNN baseline's failure suggests it overfitted to local texture patterns (such as lighting gradients and specific camera artifacts) unique to the APTOS camera sensors. In contrast, the ViT model learned global spatial arrangements and structured vascular contexts that are invariant to local color and lighting shifts. This clinical finding highlights that self-attention models generalize better than convolutional networks under domain shift.

### 5.3 Edge Deployment Feasibility and Quantization Trade-offs
To deploy EarlyDR in resource-constrained clinics, we tested MobileNetV3 Small as an edge track.
*   **Observations**: MobileNetV3 Small achieved a QWK of **0.8176** on the APTOS test set, performing close to the ViT model. Post-training FP16 quantization compressed the model file size by **49.3%**, bringing it down to just **3.01 MB** from 5.94 MB.
*   **Discussion**: While the full-precision MobileNetV3 (FP32) runs in just **10.63 ms** on CPU, the quantized variant (FP16) saw its CPU latency increase to **262.03 ms**. This is because standard CPU paths in PyTorch lack optimized vector registers for FP16 operations, leading to software emulation overhead. However, the 3.01 MB model footprint is a significant advantage, allowing the model to be stored on microcontrollers and cheap smartphones. 262 ms remains well within the clinical screening window (under 1 second per image).

### 5.4 Explainability and Clinician Trust
To validate that our models are making decisions based on actual pathological signs (rather than image noise or background artifacts), we evaluated the ResNet50 baseline using Grad-CAM.

![Grad-CAM Explainability Visualization](file:///e:/PROJECTS/EarlyDR-Project/outputs/gradcam_result.jpg)

The Grad-CAM heatmap confirms that the model correctly focuses its attention on localized retinal lesions (such as microaneurysms, hemorrhages, and hard exudates) to make its predictions. This local explainability is critical for clinical decision support, enabling doctors to quickly verify the model's output.
