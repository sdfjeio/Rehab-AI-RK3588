#!/usr/bin/env python3
"""
Convert YOLOv8-pose ONNX to RKNN for RK3588.
Run this on Google Colab or a Linux machine with rknn-toolkit2 installed.

Google Colab instructions:
1. Upload this script and yolov8n-pose.onnx to Colab
2. Run the cells below

--- Cell 1: Install rknn-toolkit2 ---
!wget -q https://github.com/airockchip/rknn-toolkit2/releases/download/v2.3.2/rknn_toolkit2-2.3.2-cp310-cp310-linux_x86_64.whl
!pip install rknn_toolkit2-2.3.2-cp310-cp310-linux_x86_64.whl

--- Cell 2: Clone model zoo for conversion scripts ---
!git clone https://github.com/airockchip/rknn_model_zoo.git
!cp yolov8n-pose.onnx rknn_model_zoo/examples/yolov8_pose/model/

--- Cell 3: Convert ---
!cd rknn_model_zoo/examples/yolov8_pose/python && python convert.py ../model/yolov8n-pose.onnx rk3588 i8

--- Cell 4: Download the converted model ---
from google.colab import files
files.download('rknn_model_zoo/examples/yolov8_pose/model/yolov8n-pose.rknn')
"""

import os
import sys
import argparse

# This script is identical to rknn_model_zoo's convert.py but adapted for standalone use.
# It requires rknn-toolkit2 to be installed: pip install rknn-toolkit2

def convert_model(onnx_path, output_path, target_platform="rk3588", quantize=True):
    """
    Convert ONNX model to RKNN format.

    Args:
        onnx_path: Path to ONNX model file
        output_path: Path for output RKNN model
        target_platform: Target platform (rk3588, rk3566, rk3568, etc.)
        quantize: Use INT8 quantization if True, FP16 otherwise
    """
    from rknn.api import RKNN

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    print(f"[1/6] Configuring model...")
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=target_platform,
        quantized_dtype='asymmetric_quantized-8' if quantize else 'float16',
    )

    # Load ONNX model
    print(f"[2/6] Loading ONNX model: {onnx_path}")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"ERROR: Failed to load ONNX model. ret={ret}")
        return ret

    # Build model
    print(f"[3/6] Building RKNN model...")
    ret = rknn.build(do_quantization=quantize, dataset='dataset.txt')
    if ret != 0:
        print(f"ERROR: Failed to build RKNN model. ret={ret}")
        return ret

    # Export RKNN model
    print(f"[4/6] Exporting RKNN model to: {output_path}")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print(f"ERROR: Failed to export RKNN model. ret={ret}")
        return ret

    # Test inference (optional sanity check)
    print(f"[5/6] Testing inference...")
    import numpy as np
    img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    outputs = rknn.inference(inputs=[img])
    if outputs:
        for i, out in enumerate(outputs):
            print(f"  Output[{i}] shape: {out.shape}, dtype: {out.dtype}")

    # Release
    print(f"[6/6] Done!")
    rknn.release()

    return 0


def main():
    parser = argparse.ArgumentParser(description="Convert YOLOv8-pose ONNX to RKNN")
    parser.add_argument("onnx", help="Path to ONNX model")
    parser.add_argument("--target", default="rk3588", help="Target platform")
    parser.add_argument("--output", help="Output RKNN path (default: <onnx_basename>.rknn)")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 instead of INT8")
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.onnx)[0]
        args.output = base + ".rknn"

    print(f"=" * 60)
    print(f"ONNX → RKNN Converter for YOLOv8-pose")
    print(f"  ONNX: {args.onnx}")
    print(f"  Output: {args.output}")
    print(f"  Target: {args.target}")
    print(f"  Quant: {'FP16' if args.fp16 else 'INT8'}")
    print(f"=" * 60)

    # Check if dataset.txt exists for quantization
    if not args.fp16 and not os.path.exists("dataset.txt"):
        print("\n[INFO] Creating dummy calibration dataset (dataset.txt)")
        # Create a simple calibration set from random images
        # For real use, replace with actual training images
        import numpy as np
        os.makedirs("calibration_images", exist_ok=True)
        with open("dataset.txt", "w") as f:
            for i in range(20):
                img_path = f"calibration_images/calib_{i:04d}.png"
                # Save a random image for calibration
                from PIL import Image
                img = Image.fromarray(np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8))
                img.save(img_path)
                f.write(img_path + "\n")
        print("[INFO] Created 20 calibration images in calibration_images/")

    ret = convert_model(args.onnx, args.output, args.target, not args.fp16)
    if ret == 0:
        print(f"\n[SUCCESS] Model converted successfully: {args.output}")
    else:
        print(f"\n[FAILED] Model conversion failed with code {ret}")
    sys.exit(ret)


if __name__ == "__main__":
    main()
