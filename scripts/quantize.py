import argparse
import os
import torch
import timm

def main():
    parser = argparse.ArgumentParser(description="Apply Dynamic Quantization to a trained PyTorch model.")
    parser.add_argument("--model", default="mobilenetv3_small_100", help="Timm model name")
    parser.add_argument("--weights_path", required=True, help="Path to the trained .pt checkpoint")
    parser.add_argument("--output_path", default="outputs/checkpoints/mobilenetv3_small_100_quantized.pt")
    args = parser.parse_args()

    print(f"Loading full-precision model: {args.model}")
    model = timm.create_model(args.model, pretrained=False, num_classes=5)
    
    # Load weights (must map to CPU since dynamic quantization is CPU-only in core PyTorch)
    model.load_state_dict(torch.load(args.weights_path, map_location=torch.device('cpu')))
    model.eval()

    # Apply FP16 (Half-Precision) Quantization
    # Note: MobileNet uses almost exclusively Conv2d layers, which PyTorch's dynamic INT8 
    # quantization ignores (it only targets Linear layers). By converting to 16-bit floats,
    # we guarantee a strict 50% memory reduction across the entire convolutional architecture.
    print("Applying FP16 (Half-Precision) compression...")
    quantized_model = model.half()

    # Save quantized model
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(quantized_model.state_dict(), args.output_path)
    
    # Compare file sizes
    original_size = os.path.getsize(args.weights_path) / (1024 * 1024)
    quantized_size = os.path.getsize(args.output_path) / (1024 * 1024)
    
    print("\n=== Quantization Results ===")
    print(f"Original Size:  {original_size:.2f} MB")
    print(f"Quantized Size: {quantized_size:.2f} MB")
    print(f"Reduction:      {(1 - quantized_size/original_size)*100:.1f}%")
    print(f"Saved to:       {args.output_path}")

if __name__ == "__main__":
    main()
