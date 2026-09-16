"""
export_onnx.py
--------------
Run this ONCE on the training PC to export your trained ResNet model to ONNX.

Usage:
    python deployment/export_onnx.py

Output:
    deployment/fire_detector.onnx  (~44 MB)
    deployment/predict_onnx.py     (copy this + the .onnx file to the other PC)
"""

import os
import sys

# onnx was installed to a short path to work around Windows MAX_PATH (260 chars)
sys.path.insert(0, r"C:\pylib")

import torch

# Add project root to path so we can import model definitions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.resnet_classifier import ResNetFireClassifier

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join("models", "saved", "best_resnet.pth")
OUTPUT_PATH = os.path.join("deployment", "fire_detector.onnx")
N_CHANNELS  = 3   # 3 for RGB images
N_CLASSES   = 1   # binary classification
INPUT_SIZE  = 224 # image will be resized to 224×224
# ──────────────────────────────────────────────────────────────────────────────


def export():
    device = torch.device("cpu")  # Always export from CPU for max compatibility

    # Load architecture
    print("Loading model architecture...")
    model = ResNetFireClassifier(n_channels=N_CHANNELS, n_classes=N_CLASSES, pretrained=False)

    # Load trained weights
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'.\n"
            "Make sure you have run training first."
        )
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded weights from: {MODEL_PATH}")

    # Create a dummy input with the correct shape: (batch=1, channels, H, W)
    dummy_input = torch.randn(1, N_CHANNELS, INPUT_SIZE, INPUT_SIZE, device=device)

    # Export to ONNX
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print(f"Exporting to ONNX -> {OUTPUT_PATH} ...")
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        export_params=True,          # Store trained weights inside the ONNX file
        opset_version=17,            # Use a modern, widely-supported opset
        do_constant_folding=True,    # Optimise constant expressions
        input_names=["image"],
        output_names=["logit"],
        dynamic_axes={               # Allow variable batch sizes at inference time
            "image": {0: "batch_size"},
            "logit": {0: "batch_size"},
        },
    )

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"\nExport complete!")
    print(f"   File : {os.path.abspath(OUTPUT_PATH)}")
    print(f"   Size : {size_mb:.1f} MB")
    print()
    print("Next steps:")
    print("  1. Copy  deployment/fire_detector.onnx  to the other PC")
    print("  2. Copy  deployment/predict_onnx.py     to the other PC")
    print("  3. On the other PC run:  pip install onnxruntime pillow numpy")
    print("  4. Run predictions:      python predict_onnx.py --image your_image.jpg")


if __name__ == "__main__":
    export()
