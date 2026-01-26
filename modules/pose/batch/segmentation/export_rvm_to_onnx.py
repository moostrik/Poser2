# type: ignore
"""Export RVM model to ONNX with TensorRT-compatible settings.

Usage:
    # Export RVM mobilenetv3 model at 256x192
    python modules/pose/batch/segmentation/export_rvm_to_onnx.py --checkpoint models/base/rvm_mobilenetv3.pth --output models/rvm_mobilenetv3_256x192.onnx

    # Export at custom resolution 384x288
    python modules/pose/batch/segmentation/export_rvm_to_onnx.py --checkpoint models/base/rvm_mobilenetv3.pth --output models/rvm_mobilenetv3_384x288.onnx --height 384 --width 288

    # Export at custom resolution 512x384
    python modules/pose/batch/segmentation/export_rvm_to_onnx.py --checkpoint models/base/rvm_mobilenetv3.pth --output models/rvm_mobilenetv3_512x384.onnx --height 512 --width 384
"""
import torch
import sys
from pathlib import Path

# Add model directory to path
sys.path.insert(0, r'C:\Developer\RobustVideoMatting')

from model import MattingNetwork


def export_rvm_to_onnx(
    checkpoint_path: str,
    output_path: str,
    height: int = 256,
    width: int = 192,
    variant: str = 'mobilenetv3',
    opset_version: int = 11,  # Lower opset avoids problematic Resize
    downsample_ratio: float = 1.0
):
    """Export RVM model to ONNX with fixed resolution (TensorRT-compatible).

    Args:
        checkpoint_path: Path to .pth checkpoint
        output_path: Output ONNX file path
        height: Fixed input height
        width: Fixed input width
        variant: Model variant ('mobilenetv3' or 'resnet50')
        opset_version: ONNX opset (11 is more TensorRT-compatible than 16+)
        downsample_ratio: RVM downsample ratio (1.0 = no downsampling)
    """
    print(f"\n{'═'*70}")
    print(f"🔄 RVM → ONNX Export (TensorRT-Compatible)")
    print(f"{'═'*70}")
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Output:      {output_path}")
    print(f"  Resolution:  {height}×{width} (fixed)")
    print(f"  Variant:     {variant}")
    print(f"  Opset:       {opset_version}")
    print(f"  Downsample:  {downsample_ratio}")
    print(f"{'═'*70}\n")

    # Load model
    print("📦 Loading model...", end='', flush=True)
    model = MattingNetwork(variant=variant, refiner='deep_guided_filter').eval()
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model = model.cuda()
    model = model.half()  # Convert to FP16 BEFORE export
    print(" ✓")

    # Create dummy inputs (also FP16)
    print("🧪 Creating dummy inputs...", end='', flush=True)
    src = torch.randn(1, 3, height, width, dtype=torch.float16).cuda()  # FP16
    r1 = None
    r2 = None
    r3 = None
    r4 = None
    downsample = torch.tensor([downsample_ratio], dtype=torch.float16).cuda()  # FP16
    print(" ✓")

    # Test forward pass
    print("🧪 Testing forward pass...", end='', flush=True)
    with torch.no_grad():
        outputs = model(src, r1, r2, r3, r4, downsample_ratio=downsample_ratio)
    print(f" ✓ (Output: {len(outputs)} tensors)")

    # Get actual recurrent state shapes from first forward pass for ONNX export
    fgr, pha, r1_out, r2_out, r3_out, r4_out = outputs

    # Export to ONNX with TensorRT-compatible settings
    print("\n💾 Exporting to ONNX...")
    print("   ⏱️  This may take 1-2 minutes...")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (src, r1_out, r2_out, r3_out, r4_out, downsample),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['src', 'r1i', 'r2i', 'r3i', 'r4i', 'downsample_ratio'],
            output_names=['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o'],
            dynamic_axes={
                # Only batch dimension is dynamic, height/width are FIXED
                'src': {0: 'batch'},
                'fgr': {0: 'batch'},
                'pha': {0: 'batch'},
            },
            verbose=False
        )

    output_file = Path(output_path)
    if output_file.exists():
        file_size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"   ✓ Export complete ({file_size_mb:.1f} MB)")
    else:
        print("   ✗ Export failed - file not created")
        return False

    print(f"\n{'═'*70}")
    print(f"✅ EXPORT COMPLETE")
    print(f"{'═'*70}")
    print(f"  Output: {output_path}")
    print(f"  Size:   {file_size_mb:.1f} MB")
    print(f"  Notes:  Fixed resolution {height}×{width} for TensorRT")
    print(f"{'═'*70}\n")

    return True


if __name__ == '__main__':
    import argparse

    print(f"\n{'═'*70}")
    print(f"🚀 RVM ONNX EXPORT TOOL")
    print(f"{'═'*70}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"{'═'*70}\n")

    parser = argparse.ArgumentParser(
        description='Export RVM PyTorch model to ONNX format',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint file (.pth)')
    parser.add_argument('--output', required=True,
                        help='Output ONNX file path (.onnx)')

    # Optional arguments with defaults
    parser.add_argument('--height', type=int, default=256,
                        help='Input image height (default: 256)')
    parser.add_argument('--width', type=int, default=192,
                        help='Input image width (default: 192)')
    parser.add_argument('--variant', default='mobilenetv3', choices=['mobilenetv3', 'resnet50'],
                        help='Model variant (default: mobilenetv3)')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset version (default: 11)')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='Downsample ratio (default: 1.0)')

    args = parser.parse_args()

    success = export_rvm_to_onnx(
        args.checkpoint,
        args.output,
        args.height,
        args.width,
        args.variant,
        args.opset,
        args.downsample
    )

    sys.exit(0 if success else 1)
