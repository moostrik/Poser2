# type: ignore
"""Export RTMPose PyTorch models to ONNX format.

Usage:
    # Export RTMPose-L 256x192 model (dimensions auto-detected from config)
    python modules/pose/batch/detection/export_rtm_to_onnx.py --config models/base/rtmpose-l_8xb256-420e_aic-coco-256x192.py --checkpoint models/base/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth --output models/rtmpose-l_256x192.onnx

    # Export RTMPose-L 384x288 model (dimensions auto-detected from config)
    python modules/pose/batch/detection/export_rtm_to_onnx.py --config models/base/rtmpose-l_8xb256-420e_aic-coco-384x288.py --checkpoint models/base/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth --output models/rtmpose-l_384x288.onnx
"""

import torch
from mmpose.apis import init_model
import argparse
import sys
from pathlib import Path
import numpy as np

# Allow numpy types in torch.load (PyTorch 2.6+ security)
torch.serialization.add_safe_globals([
    np.core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    np.dtypes.Float32DType,
    np.dtypes.UInt8DType
])


def export_rtmpose_onnx(config_file: str,
                        checkpoint_file: str,
                        output_file: str,
                        batch: int = 8,
                        opset_version: int = 11,
                        simplify: bool = True,
                        fp32: bool = False) -> bool:
    """Export RTMPose model to ONNX format.

    Args:
        config_file: Path to model config (.py)
        checkpoint_file: Path to model checkpoint (.pth)
        output_file: Output ONNX file path (.onnx)
        batch: Export batch size (enables dynamic batching 1 to batch)
        opset_version: ONNX opset version (11 recommended)
        simplify: Whether to simplify ONNX graph (requires onnx-simplifier)
        fp32: Use FP32 precision instead of FP16 (default: False)

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'═'*70}")
    print(f"🔄 RTMPose → ONNX Export")
    print(f"{'═'*70}")
    print(f"  Config:     {config_file}")
    print(f"  Checkpoint: {checkpoint_file}")
    print(f"  Output:     {output_file}")
    print(f"  Batch Size: {batch} (enables dynamic batching 1-{batch})")
    print(f"  Opset:      {opset_version}")
    print(f"  Precision:  {'FP32' if fp32 else 'FP16'}")
    print(f"  Simplify:   {'Yes' if simplify else 'No'}")
    print(f"{'═'*70}\n")

    # Validate input files
    if not Path(config_file).exists():
        print(f"❌ ERROR: Config file not found: {config_file}")
        return False
    if not Path(checkpoint_file).exists():
        print(f"❌ ERROR: Checkpoint file not found: {checkpoint_file}")
        return False

    # Load model
    print("📦 Loading model...", end='', flush=True)
    try:
        model = init_model(config_file, checkpoint_file, device='cuda:0')
        model.eval()
        if not fp32:
            model = model.half()  # Convert to FP16 before export
        print(" ✓")
    except Exception as e:
        print(" ✗")
        print(f"❌ ERROR: Failed to load model: {e}")
        return False

    # Auto-detect dimensions from model config
    print("🔍 Auto-detecting input dimensions...", end='', flush=True)
    try:
        # RTMPose stores input size in codec configuration
        codec_cfg = model.cfg.codec
        height = codec_cfg['input_size'][1]  # (width, height) format
        width = codec_cfg['input_size'][0]
        print(f" ✓ ({width}×{height})")
    except Exception as e:
        print(f" ✗")
        print(f"❌ ERROR: Failed to auto-detect dimensions: {e}")
        return False

    # Create dummy input
    print("🧪 Creating dummy input...", end='', flush=True)
    dtype = torch.float32 if fp32 else torch.float16
    dummy_input = torch.randn(batch, 3, height, width, dtype=dtype, device='cuda:0')
    print(f" ✓ (shape={dummy_input.shape}, dtype={dummy_input.dtype})")

    print(f"\n💾 Exporting to ONNX...")
    print("   ⏱️  This may take 1-2 minutes...")

    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                output_file,
                export_params=True,
                verbose=False,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['simcc_x', 'simcc_y'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'simcc_x': {0: 'batch_size'},
                    'simcc_y': {0: 'batch_size'}
                }
            )
        print("   ✓ ONNX export complete")
    except Exception as e:
        print("   ✗ Export failed")
        print(f"❌ ERROR: {e}")
        return False

    # Verify output file
    output_path = Path(output_file)
    if not output_path.exists():
        print(f"❌ ERROR: Output file not created: {output_file}")
        return False

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"   File size: {file_size_mb:.1f} MB")

    # Simplify ONNX model if requested
    if simplify:
        print("\n🔧 Simplifying ONNX model...", end='', flush=True)
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            model_onnx = onnx.load(output_file)
            model_simplified, check = onnx_simplify(model_onnx)

            if check:
                onnx.save(model_simplified, output_file)
                simplified_size_mb = output_path.stat().st_size / 1024 / 1024
                print(f" ✓ ({file_size_mb:.1f} MB → {simplified_size_mb:.1f} MB)")
            else:
                print(" ⚠️  (validation failed, keeping original)")
        except ImportError:
            print(" ⊘ (onnx-simplifier not installed)")
            print("   Install with: pip install onnx-simplifier")
        except Exception as e:
            print(f" ⚠️  (failed: {e}, keeping original)")

    # Verify ONNX model
    print("\n🔍 Verifying ONNX model...", end='', flush=True)
    try:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print(" ✓")
    except Exception as e:
        print(f" ⚠️  (validation warning: {e})")

    print(f"\n{'═'*70}")
    print(f"✅ EXPORT COMPLETE")
    print(f"{'═'*70}")
    print(f"  Output:     {output_file}")
    print(f"  Resolution: {height}×{width} (dynamic batch)")
    print(f"  Precision:  {'FP32' if fp32 else 'FP16'}")
    print(f"  Size:       {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"{'═'*70}\n")

    return True


if __name__ == '__main__':
    print(f"\n{'═'*70}")
    print(f"🚀 RTMPose ONNX EXPORT TOOL")
    print(f"{'═'*70}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"{'═'*70}\n")

    parser = argparse.ArgumentParser(
        description='Export RTMPose PyTorch model to ONNX format',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument('--config', required=True,
                        help='Path to model config file (.py)')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint file (.pth)')
    parser.add_argument('--output', required=True,
                        help='Output ONNX file path (.onnx)')

    # Optional arguments with defaults
    parser.add_argument('--batch', type=int, default=8,
                        help='Export batch size - enables dynamic batching from 1 to batch (default: 8)')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset version (default: 11)')
    parser.add_argument('--no-simplify', action='store_true',
                        help='Skip ONNX graph simplification')
    parser.add_argument('--fp32', action='store_true',
                        help='Use FP32 precision instead of FP16 (default: FP16)')

    args = parser.parse_args()

    success = export_rtmpose_onnx(
        args.config,
        args.checkpoint,
        args.output,
        args.batch,
        args.opset,
        not args.no_simplify,
        args.fp32
    )

    sys.exit(0 if success else 1)
