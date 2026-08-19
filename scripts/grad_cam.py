import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import timm
from PIL import Image
from torchvision import transforms

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to extract gradients and activations
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, target_class):
        # Forward pass
        self.model.zero_grad()
        output = self.model(x)
        
        # Backward pass for the target class
        score = output[0, target_class]
        score.backward()

        # Generate heatmap
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0) # ReLU
        if np.max(cam) > 0:
            cam = cam / np.max(cam) # Normalize
            
        return cam, output.argmax(dim=1).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, help="Path to the input fundus image")
    parser.add_argument("--model", default="resnet50", help="Timm model name (e.g., resnet50)")
    parser.add_argument("--weights_path", required=True, help="Path to the trained .pt checkpoint")
    parser.add_argument("--output_path", default="outputs/gradcam_result.jpg")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = timm.create_model(args.model, pretrained=False, num_classes=5)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.to(device)
    model.eval()

    # Identify target layer based on architecture
    # This might need adjustment depending on the exact CNN used. 
    # For ResNet, layer4 is typically the last conv block.
    if "resnet" in args.model:
        target_layer = model.layer4[-1]
    elif "efficientnet" in args.model:
        target_layer = model.blocks[-1]
    else:
        print("Grad-CAM target layer logic needs to be added for this model.")
        return

    grad_cam = GradCAM(model, target_layer)

    # Process image
    img = Image.open(args.image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Forward pass to get prediction
    with torch.no_grad():
        pred_idx = model(input_tensor).argmax(dim=1).item()
    
    print(f"Model predicted: {CLASS_NAMES[pred_idx]}")

    # Run Grad-CAM targeting the predicted class
    cam, _ = grad_cam(input_tensor, pred_idx)

    # Resize CAM to match original image
    original_img = cv2.imread(args.image_path)
    original_img = cv2.resize(original_img, (args.image_size, args.image_size))
    
    cam = cv2.resize(cam, (args.image_size, args.image_size))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    
    # Overlay heatmap
    result = heatmap * 0.5 + original_img * 0.5
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    cv2.imwrite(args.output_path, result)
    print(f"Saved Grad-CAM heatmap to {args.output_path}")

if __name__ == "__main__":
    main()
