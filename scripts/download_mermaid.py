import base64
import urllib.request
import json
import os

mermaid_code = """graph TD
    A[Raw Fundus Image] --> B[Preprocessing Module]
    B --> C{Crop & Center}
    C --> D{Resize 224x224}
    D --> E{Gaussian Blur Filter}
    E --> F[Standardized Input Tensor]
    F --> G[Deep Learning Model]
    
    subgraph DL Architecture Tracks
    G --> H[CNN Baseline: ResNet50]
    G --> I[Transformer: ViT Base]
    G --> J[Edge Model: MobileNetV3]
    end
    
    H --> K[Explainability: Grad-CAM]
    I --> L[Feature Extraction: Attention Maps]
    J --> M[Edge Compression: FP16 Quantization]
    
    K --> N[Prediction: DR Grade 0-4]
    L --> N
    M --> N
    
    N --> O((Final Output UI))
"""

state = {
    "code": mermaid_code,
    "mermaid": {"theme": "dark"}
}

json_str = json.dumps(state)
b64_str = base64.urlsafe_b64encode(json_str.encode('utf-8')).decode('utf-8')

url = f"https://mermaid.ink/img/pako:{b64_str}?type=png"
# Actually mermaid.ink uses base64 directly for simple strings, or pako for compressed.
# For json state, it's just base64.
url = f"https://mermaid.ink/img/{b64_str}?type=png"

out_path = "outputs/mermaid_flowchart.png"
print(f"Downloading from {url}")
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(out_path, 'wb') as out_file:
        out_file.write(response.read())
    print(f"Saved {out_path}")
except Exception as e:
    print(f"Failed to download: {e}")
