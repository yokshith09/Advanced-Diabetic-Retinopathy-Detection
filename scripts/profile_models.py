import os
import time
import torch
import timm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def profile_model(model_name, weights_path, device, num_warmup=10, num_runs=50):
    # Load model
    model = timm.create_model(model_name, pretrained=False, num_classes=5)
    
    # Handle quantized / FP16 model loading
    if "quantized" in weights_path:
        # Load weights on CPU first
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        # Convert to half precision (which matches FP16 quantization)
        model = model.half()
    else:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        
    model.to(device)
    model.eval()
    
    # Input tensor
    if "quantized" in weights_path:
        x = torch.randn(1, 3, 224, 224, dtype=torch.float16, device=device)
    else:
        x = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=device)
        
    params = count_parameters(model)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)
            
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_latency = ((end_time - start_time) / num_runs) * 1000 # in ms
    return params, avg_latency

def main():
    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")
    
    models_to_profile = [
        {"name": "resnet50", "weights": "outputs/checkpoints/resnet50_best.pt", "label": "ResNet50 (Baseline)"},
        {"name": "vit_base_patch16_224", "weights": "outputs/checkpoints/vit_base_patch16_224_best.pt", "label": "ViT Base"},
        {"name": "mobilenetv3_small_100", "weights": "outputs/checkpoints/mobilenetv3_small_100_best.pt", "label": "MobileNetV3 (Full-Precision)"},
        {"name": "mobilenetv3_small_100", "weights": "outputs/checkpoints/mobilenetv3_small_100_quantized.pt", "label": "MobileNetV3 (Quantized FP16)"}
    ]
    
    print("=== EarlyDR Model Profiling ===")
    print(f"Profiling on CPU and GPU (if available: {torch.cuda.is_available()})\n")
    
    for m in models_to_profile:
        if not os.path.exists(m["weights"]):
            print(f"Skipping {m['label']} - checkpoint not found.")
            continue
            
        file_size_mb = os.path.getsize(m["weights"]) / (1024 * 1024)
        
        # Profile on CPU
        try:
            # For quantized model, we only run on CPU or GPU if device matches precision
            # Quantized model is half precision, so we run on CPU using float16 if supported,
            # or just load and profile.
            params, cpu_lat = profile_model(m["name"], m["weights"], device_cpu)
        except Exception as e:
            cpu_lat = -1
            params = -1
            print(f"Error profiling {m['label']} on CPU: {e}")
            
        # Profile on GPU (if cuda is available)
        gpu_lat = -1
        if torch.cuda.is_available():
            try:
                # Quantized model (FP16) works fine on GPU
                _, gpu_lat = profile_model(m["name"], m["weights"], device_gpu)
            except Exception as e:
                gpu_lat = -1
                print(f"Error profiling {m['label']} on GPU: {e}")
                
        print(f"Model: {m['label']}")
        print(f"  Parameters:    {params:,}" if params > 0 else "  Parameters:    N/A")
        print(f"  File Size:     {file_size_mb:.2f} MB")
        print(f"  CPU Latency:   {cpu_lat:.2f} ms" if cpu_lat > 0 else "  CPU Latency:   N/A")
        print(f"  GPU Latency:   {gpu_lat:.2f} ms" if gpu_lat > 0 else "  GPU Latency:   N/A")
        print("-" * 40)

if __name__ == "__main__":
    main()
