# type: ignore
"""Export RAFT optical flow model to ONNX format.

Requires RAFT repository: https://github.com/princeton-vl/RAFT
Set PYTHONPATH to include RAFT directory before running.

Usage:
    # Export RAFT Sintel model at 256x192 (default resolution)
    python modules/pose/batch/flow/export_raft_to_onnx.py --checkpoint models/base/raft-sintel.pth --output models/raft-sintel_256x192_i12.onnx

    # Export RAFT Sintel model at 384x288
    python modules/pose/batch/flow/export_raft_to_onnx.py --checkpoint models/base/raft-sintel.pth --output models/raft-sintel_384x288_i12.onnx --height 384 --width 288

    # Export at 512x384
    python modules/pose/batch/flow/export_raft_to_onnx.py --checkpoint models/base/raft-sintel.pth --output models/raft-sintel_512x384_i12.onnx --height 512 --width 384

    # Export at 512x384 with 6 iterations
    python modules/pose/batch/flow/export_raft_to_onnx.py --checkpoint models/base/raft-sintel.pth --output models/raft-sintel_512x384_i6.onnx --height 512 --width 384 --iters 6
"""

import torch
import argparse
from pathlib import Path
import sys
import time

# Add RAFT to path (must be imported as package to support relative imports)
sys.path.insert(0, 'models/base/raft')
try:
    from core.raft import RAFT
except ImportError:
    print("❌ ERROR: Cannot import RAFT from models/base/raft/core")
    print("   Ensure RAFT core module exists at: models/base/raft/core/raft.py")
    sys.exit(1)


# ============================================================================
# ONNX EXPORT WRAPPER
# ============================================================================

class RAFTModelWrapper(torch.nn.Module):
    """Wrapper for RAFT model to handle ONNX export."""

    def __init__(self, raft_model, iters: int = 12):
        super().__init__()
        self.raft = raft_model
        self.iters = iters

    def forward(self, image1, image2):
        """Forward pass for optical flow.

        Args:
            image1: First image [B, 3, H, W] float32 in range [0, 255]
            image2: Second image [B, 3, H, W] float32 in range [0, 255]

        Returns:
            flow: Optical flow [B, 2, H, W] where flow[:, 0] is x-flow, flow[:, 1] is y-flow
        """
        # Ensure inputs match model dtype (important for FP16)
        image1 = image1.to(dtype=next(self.raft.parameters()).dtype)
        image2 = image2.to(dtype=next(self.raft.parameters()).dtype)

        # Run with test_mode=True which returns (low_res_flow, upsampled_flow)
        flow = self.raft(image1, image2, iters=self.iters, test_mode=True)
        # test_mode returns tuple: (coords1 - coords0, flow_up)
        # We want the upsampled flow (second element)
        if isinstance(flow, (tuple, list)):
            flow = flow[1]  # Take upsampled flow
        return flow


def export_raft_onnx(checkpoint_file: str, output_file: str,
                     input_height: int = 256, input_width: int = 192,
                     batch: int = 8, opset_version: int = 16,
                     simplify: bool = True, iters: int = 12, fp32: bool = True) -> bool:
    """Export RAFT model to ONNX format.

    Args:
        checkpoint_file: Path to RAFT checkpoint (.pth)
        output_file: Output ONNX file path (.onnx)
        input_height: Input image height
        input_width: Input image width
        batch: Export batch size (enables dynamic batching 1 to batch)
        opset_version: ONNX opset version (16+ required for grid_sampler)
        simplify: Whether to simplify ONNX graph (requires onnx-simplifier)
        iters: Number of RAFT refinement iterations (default: 12)
        fp32: Use FP32 precision (default: True, RAFT correlation ops are FP32-only)

    Returns:
        bool: True if successful, False otherwise
    """
    total_start = time.time()
    precision = "FP32" if fp32 else "FP16"
    dtype = torch.float32 if fp32 else torch.float16

    print(f"\n{'═'*70}")
    print(f"🔄 RAFT → ONNX Export (TensorRT-Compatible)")
    print(f"{'═'*70}")
    print(f"  Checkpoint:  {checkpoint_file}")
    print(f"  Output:      {output_file}")
    print(f"  Resolution:  {input_height}×{input_width}")
    print(f"  Batch Size:  {batch} (enables dynamic batching 1-{batch})")
    print(f"  Iterations:  {iters}")
    print(f"  Precision:   {precision}")
    print(f"  Opset:       {opset_version}")
    print(f"  Simplify:    {'Yes' if simplify else 'No'}")
    print(f"{'═'*70}")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device.upper()}")
    if device == 'cpu':
        print(f"⚠️  WARNING: CUDA not available, export will be slower")

    # Validate checkpoint file
    checkpoint_path = Path(checkpoint_file)
    if not checkpoint_path.exists():
        print(f"\n❌ ERROR: Checkpoint file not found: {checkpoint_file}")
        return False

    # Load model
    print(f"\n📦 Loading model...", end='', flush=True)
    load_start = time.time()
    try:
        from argparse import Namespace
        is_small = 'small' in checkpoint_file.lower()
        args = Namespace(small=is_small, mixed_precision=False, alternate_corr=False)

        raft_model = RAFT(args).eval()
        checkpoint = torch.load(checkpoint_file, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        raft_model.load_state_dict(state_dict)
        raft_model = raft_model.to(device)

        # Wrap model for ONNX export
        model = RAFTModelWrapper(raft_model, iters=iters)
        model.eval()

        if not fp32:
            model = model.half()

        load_time = time.time() - load_start
        file_size_mb = checkpoint_path.stat().st_size / 1024 / 1024
        print(f" ✓ ({load_time:.2f}s, {file_size_mb:.1f} MB)")
    except Exception as e:
        print(f" ✗")
        print(f"❌ ERROR: Failed to load model: {e}")
        return False

    # Create dummy inputs
    print(f"🧪 Creating dummy inputs...", end='', flush=True)
    dummy_image1 = torch.randn(batch, 3, input_height, input_width, dtype=dtype, device=device) * 255
    dummy_image2 = torch.randn(batch, 3, input_height, input_width, dtype=dtype, device=device) * 255
    print(f" ✓")

    # Test forward pass
    print(f"🧪 Testing forward pass...", end='', flush=True)
    test_start = time.time()
    try:
        with torch.no_grad():
            test_flow = model(dummy_image1, dummy_image2)
        inference_time = time.time() - test_start
        print(f" ✓ ({inference_time:.2f}s, output: {list(test_flow.shape)})")
    except Exception as e:
        print(f" ✗")
        print(f"❌ ERROR: Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Export to ONNX
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Exporting to ONNX...")
    print(f"   ⏱️  This may take 1-2 minutes...")
    export_start = time.time()
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_image1, dummy_image2),
                output_file,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['image1', 'image2'],
                output_names=['flow'],
                dynamic_axes={
                    'image1': {0: 'batch_size'},
                    'image2': {0: 'batch_size'},
                    'flow': {0: 'batch_size'}
                },
                verbose=False
            )
        export_time = time.time() - export_start
        print(f"   ✓ ONNX export complete ({export_time:.2f}s)")
    except Exception as e:
        print(f"   ✗ Export failed")
        print(f"❌ ERROR: {e}")
        return False

    if not output_path.exists():
        print(f"❌ ERROR: Output file was not created!")
        return False

    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"   File size: {file_size:.1f} MB")

    # Simplify ONNX graph
    if simplify:
        print(f"\n🔧 Simplifying ONNX model...", end='', flush=True)
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            model_onnx = onnx.load(output_file)
            model_simplified, check = onnx_simplify(model_onnx)

            if check:
                onnx.save(model_simplified, output_file)
                new_size = output_path.stat().st_size / 1024 / 1024
                reduction = ((file_size - new_size) / file_size) * 100
                print(f" ✓ ({file_size:.1f} MB → {new_size:.1f} MB, {reduction:.1f}% reduction)")
            else:
                print(f" ⚠️  (validation failed, keeping original)")
        except ImportError:
            print(f" ⊘ (onnx-simplifier not installed)")
            print(f"   Install with: pip install onnx-simplifier")
        except Exception as e:
            print(f" ⚠️  (failed: {e}, keeping original)")

    # Verify ONNX model
    print(f"\n🔍 Verifying ONNX model...", end='', flush=True)
    try:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print(f" ✓")
        print(f"   Inputs:  {[i.name for i in onnx_model.graph.input]}")
        print(f"   Outputs: {[o.name for o in onnx_model.graph.output]}")
    except Exception as e:
        print(f" ⚠️  (validation warning: {e})")

    total_time = time.time() - total_start
    print(f"\n{'═'*70}")
    print(f"✅ EXPORT COMPLETE")
    print(f"{'═'*70}")
    print(f"  Output:     {output_file}")
    print(f"  Resolution: {input_height}×{input_width} (dynamic batch)")
    print(f"  Precision:  {precision}")
    print(f"  Size:       {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Total time: {total_time:.2f}s")
    print(f"{'═'*70}\n")

    return True


if __name__ == '__main__':
    print(f"\n{'═'*70}")
    print(f"🚀 RAFT ONNX EXPORT TOOL")
    print(f"{'═'*70}")
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA:    {'✓ ' + torch.version.cuda if torch.cuda.is_available() else '✗ Not available'}")
    if torch.cuda.is_available():
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")
    print(f"{'═'*70}\n")

    parser = argparse.ArgumentParser(
        description='Export RAFT optical flow model to ONNX',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument('--checkpoint', required=True,
                        help='Path to RAFT checkpoint file (.pth)')
    parser.add_argument('--output', required=True,
                        help='Output ONNX file path (.onnx)')

    # Optional arguments with defaults
    parser.add_argument('--height', type=int, default=256,
                        help='Input image height (default: 256)')
    parser.add_argument('--width', type=int, default=192,
                        help='Input image width (default: 192)')
    parser.add_argument('--batch', type=int, default=8,
                        help='Export batch size - enables dynamic batching from 1 to batch (default: 8)')
    parser.add_argument('--iters', type=int, default=12,
                        help='RAFT refinement iterations (default: 12)')
    parser.add_argument('--opset', type=int, default=16,
                        help='ONNX opset version (default: 16, required for grid_sampler)')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 precision instead of FP32 (default: FP32, may have issues)')
    parser.add_argument('--no-simplify', action='store_true',
                        help='Skip ONNX graph simplification')

    args = parser.parse_args()

    success = export_raft_onnx(
        args.checkpoint,
        args.output,
        args.height,
        args.width,
        args.batch,
        args.opset,
        not args.no_simplify,
        args.iters,
        not args.fp16  # Invert: fp16 flag means fp32=False
    )

    sys.exit(0 if success else 1)