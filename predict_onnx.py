"""
predict_onnx.py
---------------
Standalone fire detection predictor — NO PyTorch required.

Copy ONLY these two files to the other PC:
    - fire_detector.onnx
    - predict_onnx.py

Install dependencies (one-time):
    pip install onnxruntime pillow numpy

Usage:
    # Single image
    python predict_onnx.py --image path/to/image.jpg

    # Folder of images
    python predict_onnx.py --folder path/to/images/

    # Custom model path
    python predict_onnx.py --image fire.jpg --model path/to/fire_detector.onnx

    # Adjust decision threshold (default 0.5)
    python predict_onnx.py --image fire.jpg --threshold 0.4
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_MODEL  = os.path.join(os.path.dirname(__file__), "fire_detector.onnx")
INPUT_SIZE     = 224
# ImageNet normalization (matches training preprocessing)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# ──────────────────────────────────────────────────────────────────────────────


def load_session(model_path: str):
    """Load the ONNX model into an inference session."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime is not installed.")
        print("Run:  pip install onnxruntime")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Make sure 'fire_detector.onnx' is in the same folder as this script.")
        sys.exit(1)

    # Prefer GPU (CUDA) if available, fall back to CPU
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
        print("[GPU] Running on CUDA")
    else:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print("[CPU] Running on CPU")

    return session


def preprocess(image_path: str) -> np.ndarray:
    """Load and preprocess an image to match training pipeline."""
    img = Image.open(image_path).convert("RGB")

    # Resize short side to 256, then centre-crop to 224×224
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    left  = (img.width  - INPUT_SIZE) // 2
    upper = (img.height - INPUT_SIZE) // 2
    img = img.crop((left, upper, left + INPUT_SIZE, upper + INPUT_SIZE))

    # Convert to float32 array in [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)

    # Normalize with ImageNet stats
    arr = (arr - MEAN) / STD

    # NHWC → NCHW, add batch dimension
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 224, 224)
    return arr.astype(np.float32)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def predict_one(session, image_path: str, threshold: float = 0.5) -> dict:
    """Run inference on a single image. Returns a result dict."""
    input_array = preprocess(image_path)

    input_name = session.get_inputs()[0].name
    t0 = time.perf_counter()
    logit = session.run(None, {input_name: input_array})[0][0, 0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    prob = float(sigmoid(logit))
    is_fire = prob >= threshold

    return {
        "file":             os.path.basename(image_path),
        "fire_probability": prob,
        "is_fire":          is_fire,
        "inference_ms":     elapsed_ms,
    }


def print_result(result: dict, threshold: float):
    label = "[FIRE]    " if result["is_fire"] else "[NO FIRE] "
    print(
        f"  {label}  |  "
        f"Probability: {result['fire_probability']:.1%}  |  "
        f"Time: {result['inference_ms']:.1f} ms  |  "
        f"{result['file']}"
    )


def run_single(session, image_path: str, threshold: float):
    print(f"\nRunning prediction on: {image_path}")
    result = predict_one(session, image_path, threshold)
    print_result(result, threshold)
    return result


def run_folder(session, folder_path: str, threshold: float):
    folder = Path(folder_path)
    images = [p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTS]

    if not images:
        print(f"No supported images found in: {folder_path}")
        return

    print(f"\nScanning {len(images)} image(s) in: {folder_path}\n")
    print(f"{'File':<40} {'Probability':>12}  {'Label':<12}  {'ms':>8}")
    print("-" * 78)

    fire_count = 0
    for img_path in sorted(images):
        try:
            result = predict_one(session, str(img_path), threshold)
            label = "[FIRE]" if result["is_fire"] else "[NO FIRE]"
            print(
                f"  {result['file']:<38} "
                f"{result['fire_probability']:>11.1%}  "
                f"{label:<12}  "
                f"{result['inference_ms']:>6.1f} ms"
            )
            if result["is_fire"]:
                fire_count += 1
        except Exception as exc:
            print(f"  [ERR] {img_path.name:<38} ERROR: {exc}")

    print("-" * 78)
    print(f"  Summary: {fire_count}/{len(images)} image(s) detected as FIRE  "
          f"(threshold={threshold:.2f})\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fire detection inference using ONNX — no PyTorch required."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Path to a single image file")
    group.add_argument("--folder", type=str, help="Path to a folder of images")

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Path to fire_detector.onnx (default: same directory as this script)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for fire classification (default: 0.5)",
    )
    args = parser.parse_args()

    print("\nWildfire Detection - ONNX Inference")
    print(f"   Model     : {args.model}")
    print(f"   Threshold : {args.threshold}")

    session = load_session(args.model)

    if args.image:
        run_single(session, args.image, args.threshold)
    else:
        run_folder(session, args.folder, args.threshold)


if __name__ == "__main__":
    main()
