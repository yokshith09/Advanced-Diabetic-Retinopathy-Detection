import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_resnet_block(ax):
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Title
    ax.text(5, 9.5, "ResNet50 Bottleneck Residual Block", fontsize=12, fontweight='bold', ha='center')
    
    # Draw boxes
    # Input
    ax.text(5, 8.5, "Input (x)", bbox=dict(boxstyle="round,pad=0.3", fc="#f5f6fa", ec="#2f3640", lw=1.5), ha='center')
    
    # Conv 1x1
    ax.text(5, 7.0, "Conv 1x1\n(Reduce Channels)", bbox=dict(boxstyle="square,pad=0.5", fc="#dcdde1", ec="#2f3640", lw=1.5), ha='center')
    
    # Conv 3x3
    ax.text(5, 5.2, "Conv 3x3\n(Spatial Feature Extraction)", bbox=dict(boxstyle="square,pad=0.5", fc="#3498db", ec="#2f3640", lw=1.5), ha='center', color='white')
    
    # Conv 1x1
    ax.text(5, 3.4, "Conv 1x1\n(Restore Channels)", bbox=dict(boxstyle="square,pad=0.5", fc="#dcdde1", ec="#2f3640", lw=1.5), ha='center')
    
    # Add block
    ax.text(5, 1.8, "( + ) Addition", bbox=dict(boxstyle="circle,pad=0.3", fc="#e74c3c", ec="#2f3640", lw=1.5), ha='center', color='white')
    
    # Output
    ax.text(5, 0.5, "Output: ReLU(F(x) + x)", bbox=dict(boxstyle="round,pad=0.3", fc="#f5f6fa", ec="#2f3640", lw=1.5), ha='center')
    
    # Draw arrows
    # Main path
    ax.annotate('', xy=(5, 7.5), xytext=(5, 8.2), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate('', xy=(5, 5.8), xytext=(5, 6.4), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate('', xy=(5, 4.0), xytext=(5, 4.6), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate('', xy=(5, 2.2), xytext=(5, 2.8), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate('', xy=(5, 0.8), xytext=(5, 1.4), arrowprops=dict(arrowstyle="->", lw=1.5))
    
    # Skip connection path
    # Draw a line from input, around, and to addition
    skip_path = patches.FancyArrowPatch((5, 8.2), (5, 2.0),
                                        connectionstyle="arc3,rad=-0.8",
                                        arrowstyle="->",
                                        mutation_scale=15,
                                        lw=1.5,
                                        linestyle='--',
                                        color='#7f8c8d')
    ax.add_patch(skip_path)
    ax.text(8.0, 5.2, "Skip Connection\n(Identity / Conv 1x1)", fontsize=9, color='#7f8c8d', ha='center')

def draw_vit_pipeline(ax):
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Title
    ax.text(5, 9.5, "Vision Transformer (ViT) Pipeline", fontsize=12, fontweight='bold', ha='center')
    
    # Draw pipeline boxes
    ax.text(5, 8.6, "Input Retinal Image (224x224)", bbox=dict(boxstyle="round,pad=0.3", fc="#f5f6fa", ec="#2f3640", lw=1.5), ha='center')
    ax.text(5, 7.4, "Patch Extraction (16x16) & Flattening", bbox=dict(boxstyle="square,pad=0.4", fc="#ffeaa7", ec="#2f3640", lw=1.5), ha='center')
    ax.text(5, 6.2, "Linear Projection + Position Embeddings", bbox=dict(boxstyle="square,pad=0.4", fc="#dcdde1", ec="#2f3640", lw=1.5), ha='center')
    ax.text(5, 4.8, "Transformer Encoder (x12)\n• Multi-Head Self-Attention (MHSA)\n• MLP Blocks with GELU\n• Layer Normalization", bbox=dict(boxstyle="square,pad=0.5", fc="#3498db", ec="#2f3640", lw=1.5), ha='center', color='white')
    ax.text(5, 3.2, "MLP Head (Classification Layer)", bbox=dict(boxstyle="square,pad=0.4", fc="#ffeaa7", ec="#2f3640", lw=1.5), ha='center')
    ax.text(5, 2.0, "Multi-class Output (DR Grade 0-4)", bbox=dict(boxstyle="round,pad=0.3", fc="#f5f6fa", ec="#2f3640", lw=1.5), ha='center')
    
    # Draw arrows
    ax.annotate('', xy=(5, 7.8), xytext=(5, 8.3), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate('', xy=(5, 6.6), xytext=(5, 7.0), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate('', xy=(5, 5.4), xytext=(5, 5.8), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate('', xy=(5, 3.6), xytext=(5, 4.2), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate('', xy=(5, 2.4), xytext=(5, 2.8), arrowprops=dict(arrowstyle="->", lw=1.5))

def draw_mobilenet_block(ax):
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Title
    ax.text(5, 9.5, "MobileNetV3 Inverted Residual Block + SE", fontsize=12, fontweight='bold', ha='center')
    
    # Draw boxes
    ax.text(5, 8.6, "Input (x)", bbox=dict(boxstyle="round,pad=0.3", fc="#f5f6fa", ec="#2f3640", lw=1.5), ha='center')
    ax.text(5, 7.4, "Conv 1x1 (Expansion to higher dim)", bbox=dict(boxstyle="square,pad=0.4", fc="#dcdde1", ec="#2f3640", lw=1.5), ha='center')
    ax.text(5, 6.2, "Depthwise Conv 3x3 / 5x5 (Lightweight)", bbox=dict(boxstyle="square,pad=0.4", fc="#2ecc71", ec="#2f3640", lw=1.5), ha='center', color='white')
    ax.text(5, 4.8, "Squeeze-and-Excitation (SE) Block\n(Global Pooling + Two FC Layers + Sigmoid)", bbox=dict(boxstyle="square,pad=0.4", fc="#9b59b6", ec="#2f3640", lw=1.5), ha='center', color='white')
    ax.text(5, 3.4, "Conv 1x1 (Linear Projection to lower dim)", bbox=dict(boxstyle="square,pad=0.4", fc="#dcdde1", ec="#2f3640", lw=1.5), ha='center')
    ax.text(5, 2.0, "( + ) Addition", bbox=dict(boxstyle="circle,pad=0.3", fc="#e74c3c", ec="#2f3640", lw=1.5), ha='center', color='white')
    ax.text(5, 0.8, "Output", bbox=dict(boxstyle="round,pad=0.3", fc="#f5f6fa", ec="#2f3640", lw=1.5), ha='center')
    
    # Draw arrows
    ax.annotate('', xy=(5, 7.8), xytext=(5, 8.3), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate('', xy=(5, 6.6), xytext=(5, 7.0), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate('', xy=(5, 5.4), xytext=(5, 5.8), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate('', xy=(5, 3.8), xytext=(5, 4.4), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate('', xy=(5, 2.4), xytext=(5, 3.0), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate('', xy=(5, 1.2), xytext=(5, 1.6), arrowprops=dict(arrowstyle="->", lw=1.5))
    
    # Skip connection path
    skip_path = patches.FancyArrowPatch((5, 8.3), (5, 2.2),
                                        connectionstyle="arc3,rad=-0.8",
                                        arrowstyle="->",
                                        mutation_scale=15,
                                        lw=1.5,
                                        linestyle='--',
                                        color='#7f8c8d')
    ax.add_patch(skip_path)
    ax.text(8.0, 5.3, "Shortcut (Skip)\n(Residual Connection)", fontsize=9, color='#7f8c8d', ha='center')

def main():
    fig, axes = plt.subplots(1, 3, figsize=(18, 9))
    
    draw_resnet_block(axes[0])
    draw_vit_pipeline(axes[1])
    draw_mobilenet_block(axes[2])
    
    plt.tight_layout()
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "deep_learning_architectures.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved deep_learning_architectures.png")

if __name__ == "__main__":
    main()
