# type: ignore
"""Export RAFT optical flow model to ONNX format.

Requires RAFT repository: https://github.com/princeton-vl/RAFT
Set PYTHONPATH to include RAFT directory before running.
"""

import torch
import argparse
from pathlib import Path
import sys
import time


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
        # Run with test_mode=True which returns (low_res_flow, upsampled_flow)
        flow = self.raft(image1, image2, iters=self.iters, test_mode=True)
        # test_mode returns tuple: (coords1 - coords0, flow_up)
        # We want the upsampled flow (second element)
        if isinstance(flow, (tuple, list)):
            flow = flow[1]  # Take upsampled flow
        return flow


def load_raft_model(checkpoint_path: str, device: str = 'cuda:0', small: bool = False):
    """Load RAFT model from checkpoint.

    Args:
        checkpoint_path: Path to RAFT .pth checkpoint
        device: Device to load model on
        small: Whether to use RAFT-Small architecture

    Returns:
        Loaded RAFT model
    """
    print(f"\n{'─'*60}")
    print(f"📦 LOADING MODEL")
    print(f"{'─'*60}")

    # Add RAFT to path
    print(f"  ├─ Adding RAFT to Python path...", end='', flush=True)
    raft_path = Path("c:/Developer/RAFT")
    raft_core_path = raft_path / "core"

    if not raft_path.exists():
        print(f" ✗")
        print(f"\n❌ ERROR: RAFT directory not found at {raft_path}")
        print(f"   Clone with: git clone https://github.com/princeton-vl/RAFT.git c:/Developer/RAFT")
        sys.exit(1)

    # Add both RAFT root and core directory
    if str(raft_path) not in sys.path:
        sys.path.insert(0, str(raft_path))
    if str(raft_core_path) not in sys.path:
        sys.path.insert(0, str(raft_core_path))
    print(f"  ├─ Importing RAFT from repository...", end='', flush=True)
    try:
        from core.raft import RAFT
        print(f" ✓")
    except ImportError as e:
        print(f" ✗")
        print(f"\n❌ ERROR: Cannot import RAFT")
        print(f"   Error: {e}")
        print(f"\n   Diagnostics:")
        print(f"   - RAFT path exists: {raft_path.exists()}")
        print(f"   - RAFT path added to sys.path: {str(raft_path) in sys.path}")
        print(f"   - Looking for: {raft_path / 'core' / 'raft.py'}")
        print(f"   - File exists: {(raft_path / 'core' / 'raft.py').exists()}")
        print(f"   - Directory contents:")
        if raft_path.exists():
            for item in sorted(raft_path.iterdir())[:10]:
                print(f"      - {item.name}")
        print(f"\n   Try running in terminal:")
        print(f"   cd c:\\Developer\\RAFT")
        print(f"   dir core\\raft.py")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        print(f"\n   Current sys.path:")
        for p in sys.path[:5]:
            print(f"     - {p}")
        if len(sys.path) > 5:
            print(f"     ... ({len(sys.path)-5} more)")
        sys.exit(1)

    # Create model
    print(f"  ├─ Creating RAFT architecture...", end='', flush=True)
    start = time.time()

    # Create args object for RAFT using argparse.Namespace for compatibility
    from argparse import Namespace
    args = Namespace(
        small=small,
        mixed_precision=False,
        alternate_corr=False
    )

    model = RAFT(args)
    print(f" ✓ ({time.time()-start:.2f}s)")

    # Verify checkpoint file
    print(f"  ├─ Checking checkpoint file...", end='', flush=True)
    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        print(f" ✗")
        print(f"\n❌ ERROR: Checkpoint file not found!")
        print(f"   Path: {checkpoint_path_obj.absolute()}")
        sys.exit(1)
    file_size_mb = checkpoint_path_obj.stat().st_size / 1024 / 1024
    print(f" ✓ ({file_size_mb:.1f} MB)")

    # Load checkpoint
    print(f"  ├─ Loading checkpoint...", end='', flush=True)
    start = time.time()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_time = time.time() - start
    print(f" ✓ ({load_time:.2f}s)")

    # Extract state dict
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"  │  └─ Using checkpoint['state_dict']")
        else:
            state_dict = checkpoint
            print(f"  │  └─ Using checkpoint directly ({list(checkpoint.keys())[0] if checkpoint else 'empty'}...)")
    else:
        state_dict = checkpoint
        print(f"  │  └─ Direct state dict")

    # Clean keys
    original_key = list(state_dict.keys())[0] if state_dict else ''
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    new_key = list(state_dict.keys())[0] if state_dict else ''

    if original_key != new_key:
        print(f"  │  └─ Removed 'module.' prefix from keys")

    print(f"  │     ({len(state_dict)} parameters)")

    # Load weights into model
    print(f"  ├─ Loading weights into model...", end='', flush=True)
    start = time.time()
    try:
        model.load_state_dict(state_dict)
        print(f" ✓ ({time.time()-start:.2f}s)")
    except Exception as e:
        print(f" ✗")
        print(f"\n❌ ERROR: Failed to load weights: {e}")
        raise

    # Move to device
    print(f"  └─ Moving to {device}...", end='', flush=True)
    start = time.time()
    model = model.to(device)
    model.eval()
    print(f" ✓ ({time.time()-start:.2f}s)")

    print(f"{'─'*60}")
    print(f"✓ Model loaded successfully")
    print(f"{'─'*60}")

    return model


def export_raft_onnx(checkpoint_file: str, output_file: str,
                     input_height: int = 256, input_width: int = 192,
                     opset_version: int = 16, simplify: bool = True, iters: int = 12):
    """Export RAFT model to ONNX format.

    Args:
        checkpoint_file: Path to RAFT checkpoint (.pth)
        output_file: Output ONNX file path (.onnx)
        input_height: Input image height
        input_width: Input image width
        opset_version: ONNX opset version (16+ required for grid_sampler)
        simplify: Whether to simplify ONNX graph (requires onnx-simplifier)
        iters: Number of RAFT refinement iterations (default: 12)
    """
    total_start = time.time()

    print(f"\n{'═'*60}")
    print(f"🔄 RAFT → ONNX EXPORT")
    print(f"{'═'*60}")
    print(f"  Checkpoint:  {checkpoint_file}")
    print(f"  Output:      {output_file}")
    print(f"  Resolution:  {input_height}×{input_width}")
    print(f"  Iterations:  {iters}")
    print(f"  Opset:       {opset_version}")
    print(f"  Simplify:    {'Yes' if simplify else 'No'}")
    print(f"{'═'*60}")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device.upper()}")
    if device == 'cpu':
        print(f"⚠️  WARNING: CUDA not available, export will be slower")

    # Step 1: Load model
    print(f"\n[1/4] 📥 LOADING MODEL")
    is_small = 'small' in checkpoint_file.lower()
    raft_model = load_raft_model(checkpoint_file, device, small=is_small)

    # Step 2: Wrap model
    print(f"\n[2/4] 📦 PREPARING FOR ONNX EXPORT")
    print(f"  └─ Wrapping model (iters={iters})...", end='', flush=True)
    start = time.time()
    model = RAFTModelWrapper(raft_model, iters=iters)
    model.eval()
    print(f" ✓ ({time.time()-start:.2f}s)")

    # Step 3: Test forward pass
    print(f"\n[3/4] 🧪 TESTING FORWARD PASS")
    print(f"  ├─ Creating dummy inputs ({input_height}×{input_width})...", end='', flush=True)
    start = time.time()
    dummy_image1 = torch.randn(1, 3, input_height, input_width, device=device) * 255
    dummy_image2 = torch.randn(1, 3, input_height, input_width, device=device) * 255
    print(f" ✓ ({time.time()-start:.3f}s)")

    print(f"  └─ Running test inference...", end='', flush=True)
    start = time.time()
    try:
        with torch.no_grad():
            test_flow = model(dummy_image1, dummy_image2)
        inference_time = time.time() - start
        print(f" ✓ ({inference_time:.2f}s)")
        print(f"     └─ Output shape: {list(test_flow.shape)} ({test_flow.dtype})")
    except Exception as e:
        print(f" ✗")
        print(f"\n❌ ERROR: Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Step 4: Export to ONNX
    print(f"\n[4/4] 💾 EXPORTING TO ONNX")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  └─ Running torch.onnx.export...", end='', flush=True)
    start = time.time()
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
        export_time = time.time() - start
        print(f" ✓ ({export_time:.2f}s)")
    except Exception as e:
        print(f" ✗")
        print(f"\n❌ ERROR: ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    if not output_path.exists():
        print(f"\n❌ ERROR: Output file was not created!")
        sys.exit(1)

    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"     └─ File size: {file_size:.2f} MB")

    # Optional: Simplify ONNX graph
    if simplify:
        print(f"\n🔧 SIMPLIFYING ONNX GRAPH")
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            print(f"  ├─ Loading ONNX model...", end='', flush=True)
            start = time.time()
            model_onnx = onnx.load(output_file)
            print(f" ✓ ({time.time()-start:.2f}s)")

            print(f"  ├─ Running simplification...", end='', flush=True)
            start = time.time()
            model_simplified, check = onnx_simplify(model_onnx)
            simplify_time = time.time() - start
            print(f" ✓ ({simplify_time:.2f}s)")

            if check:
                print(f"  └─ Saving simplified model...", end='', flush=True)
                start = time.time()
                onnx.save(model_simplified, output_file)
                print(f" ✓ ({time.time()-start:.2f}s)")

                new_size = output_path.stat().st_size / 1024 / 1024
                reduction = ((file_size - new_size) / file_size) * 100
                print(f"     └─ New size: {new_size:.2f} MB ({reduction:.1f}% reduction)")
            else:
                print(f"  └─ ⚠️  Simplification check failed, keeping original")
        except ImportError:
            print(f"  └─ ⚠️  onnx-simplifier not installed")
            print(f"     Install with: pip install onnx-simplifier")
        except Exception as e:
            print(f"  └─ ⚠️  Simplification failed: {e}")
            print(f"     Keeping original file")

    # Verify ONNX model
    print(f"\n✅ VERIFYING ONNX MODEL")
    try:
        import onnx
        print(f"  ├─ Loading and checking model...", end='', flush=True)
        start = time.time()
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print(f" ✓ ({time.time()-start:.2f}s)")

        print(f"  └─ Model information:")
        print(f"     ├─ Inputs:  {[i.name for i in onnx_model.graph.input]}")
        print(f"     ├─ Outputs: {[o.name for o in onnx_model.graph.output]}")
        print(f"     └─ Path:    {output_path.absolute()}")
    except Exception as e:
        print(f" ✗")
        print(f"  └─ ⚠️  Verification failed: {e}")

    total_time = time.time() - total_start
    print(f"\n{'═'*60}")
    print(f"✅ EXPORT COMPLETED SUCCESSFULLY")
    print(f"{'═'*60}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Output:     {output_path.absolute()}")
    print(f"{'═'*60}\n")


def export_all_raft_models(models_dir: str = "models"):
    """Export all RAFT models to ONNX."""
    print(f"\n{'═'*60}")
    print(f"📦 BATCH EXPORT MODE")
    print(f"{'═'*60}")

    models = [
        # (checkpoint, output, height, width)
        # ('raft-sintel.pth', 'raft-sintel_256x192.onnx', 256, 192),
        # ('raft-sintel.pth', 'raft-sintel_384x288.onnx', 384, 288),
        # ('raft-sintel.pth', 'raft-sintel_512x384.onnx', 512, 384),
        ('raft-small.pth', 'raft-small_256x192.onnx', 256, 192),
        # ('raft-things.pth', 'raft-things_256x192.onnx', 256, 192),
    ]

    models_path = Path(models_dir)
    print(f"  Models directory: {models_path.absolute()}")
    print(f"  Total models:     {len(models)}")
    print(f"{'═'*60}\n")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for idx, (checkpoint, output, height, width) in enumerate(models, 1):
        checkpoint_path = models_path / checkpoint
        output_path = models_path / output

        print(f"\n{'─'*60}")
        print(f"[{idx}/{len(models)}] Processing: {output}")
        print(f"{'─'*60}")

        if not checkpoint_path.exists():
            print(f"⊘ Skipping: {checkpoint} not found")
            print(f"   Path: {checkpoint_path}")
            skip_count += 1
            continue

        try:
            export_raft_onnx(str(checkpoint_path), str(output_path), height, width)
            success_count += 1
        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    # Summary
    print(f"\n{'═'*60}")
    print(f"📊 BATCH EXPORT SUMMARY")
    print(f"{'═'*60}")
    print(f"  ✅ Successful: {success_count}")
    print(f"  ⊘  Skipped:    {skip_count}")
    print(f"  ❌ Failed:     {fail_count}")
    print(f"  ━  Total:      {len(models)}")
    print(f"{'═'*60}\n")


if __name__ == '__main__':
    print(f"\n{'═'*60}")
    print(f"🚀 RAFT ONNX EXPORT TOOL")
    print(f"{'═'*60}")
    print(f"  Python:       {sys.version.split()[0]}")
    print(f"  PyTorch:      {torch.__version__}")
    print(f"  CUDA:         {'✓ ' + torch.version.cuda if torch.cuda.is_available() else '✗ Not available'}")
    if torch.cuda.is_available():
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
    print(f"{'═'*60}\n")

    parser = argparse.ArgumentParser(description='Export RAFT optical flow model to ONNX')
    parser.add_argument('--models-dir', default='models', help='Directory containing models')
    parser.add_argument('--checkpoint', help='RAFT checkpoint file (.pth)')
    parser.add_argument('--output', help='Output ONNX file (.onnx)')
    parser.add_argument('--height', type=int, default=256, help='Input height (default: 256)')
    parser.add_argument('--width', type=int, default=192, help='Input width (default: 192)')
    parser.add_argument('--iters', type=int, default=12, help='RAFT refinement iterations (default: 12)')
    parser.add_argument('--opset', type=int, default=16, help='ONNX opset version (default: 16, required for grid_sampler)')
    parser.add_argument('--no-simplify', action='store_true', help='Skip ONNX simplification')

    args = parser.parse_args()

    if args.checkpoint and args.output:
        # Export single model
        print(f"Mode: Single model export\n")
        try:
            export_raft_onnx(
                args.checkpoint,
                args.output,
                args.height,
                args.width,
                args.opset,
                not args.no_simplify,
                args.iters
            )
        except Exception as e:
            print(f"\n{'═'*60}")
            print(f"❌ EXPORT FAILED")
            print(f"{'═'*60}")
            print(f"  Error: {e}")
            print(f"{'═'*60}\n")
            sys.exit(1)
    else:
        # Export all models
        print(f"Mode: Batch export\n")
        export_all_raft_models(args.models_dir)