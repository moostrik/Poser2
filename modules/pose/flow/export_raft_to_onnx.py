# type: ignore
"""Export RAFT optical flow model to ONNX format."""

import torch
import argparse
from pathlib import Path
import sys


class RAFTModelWrapper(torch.nn.Module):
    """Wrapper for RAFT model to handle ONNX export."""

    def __init__(self, raft_model):
        super().__init__()
        self.raft = raft_model

    def forward(self, image1, image2):
        """Forward pass for optical flow.

        Args:
            image1: First image [B, 3, H, W] float32 in range [0, 255]
            image2: Second image [B, 3, H, W] float32 in range [0, 255]

        Returns:
            flow: Optical flow [B, 2, H, W] where flow[:, 0] is x-flow, flow[:, 1] is y-flow
        """
        # RAFT expects images in [0, 255] range
        # Run with fixed iterations for ONNX compatibility
        flow_predictions = self.raft(image1, image2, iters=12, test_mode=True)

        # Return final flow prediction
        return flow_predictions[-1]


def load_raft_model(checkpoint_path: str, device: str = 'cuda:0'):
    """Load RAFT model from checkpoint.

    Args:
        checkpoint_path: Path to RAFT .pth checkpoint
        device: Device to load model on

    Returns:
        Loaded RAFT model
    """
    print(f"\n[DEBUG] Attempting to import RAFT...")
    # Add RAFT directory to path if needed
    try:
        from core.raft import RAFT
        from core.utils.utils import InputPadder
        print(f"[DEBUG] ✓ RAFT imported successfully")
    except ImportError as e:
        print(f"[DEBUG] ✗ RAFT import failed: {e}")
        print("\nERROR: RAFT not found in Python path")
        print("Clone RAFT repo and add to path:")
        print("  git clone https://github.com/princeton-vl/RAFT.git")
        print("  set PYTHONPATH=%PYTHONPATH%;c:\\Developer\\RAFT")
        print("\nCurrent sys.path:")
        for p in sys.path:
            print(f"  - {p}")
        sys.exit(1)

    print(f"[DEBUG] Creating RAFT model instance...")
    # Create RAFT model (default args)
    class Args:
        small = False
        mixed_precision = False
        alternate_corr = False

    args = Args()
    model = RAFT(args)
    print(f"[DEBUG] ✓ RAFT model created")

    # Load checkpoint
    print(f"\n[DEBUG] Loading checkpoint from: {checkpoint_path}")
    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        print(f"[DEBUG] ✗ Checkpoint file not found!")
        print(f"[DEBUG] Checked path: {checkpoint_path_obj.absolute()}")
        sys.exit(1)

    print(f"[DEBUG] File size: {checkpoint_path_obj.stat().st_size / 1024 / 1024:.2f} MB")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"[DEBUG] ✓ Checkpoint loaded")
    print(f"[DEBUG] Checkpoint type: {type(checkpoint)}")
    if isinstance(checkpoint, dict):
        print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys())}")

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"[DEBUG] Using checkpoint['state_dict']")
    else:
        state_dict = checkpoint
        print(f"[DEBUG] Using checkpoint directly as state_dict")

    print(f"[DEBUG] State dict has {len(state_dict)} keys")

    # Remove 'module.' prefix if present (from DataParallel)
    original_keys = list(state_dict.keys())[:3]
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    new_keys = list(state_dict.keys())[:3]
    if original_keys != new_keys:
        print(f"[DEBUG] Removed 'module.' prefix from keys")
    print(f"[DEBUG] First 3 keys: {new_keys}")

    print(f"[DEBUG] Loading state dict into model...")
    try:
        model.load_state_dict(state_dict)
        print(f"[DEBUG] ✓ State dict loaded successfully")
    except Exception as e:
        print(f"[DEBUG] ✗ Failed to load state dict: {e}")
        raise

    print(f"[DEBUG] Moving model to device: {device}")
    model = model.to(device)
    model.eval()
    print(f"[DEBUG] ✓ Model ready for export")

    return model


def export_raft_onnx(checkpoint_file: str, output_file: str,
                     input_height: int = 256, input_width: int = 192,
                     opset_version: int = 11, simplify: bool = True):
    """Export RAFT model to ONNX format.

    Args:
        checkpoint_file: Path to RAFT checkpoint (.pth)
        output_file: Output ONNX file path (.onnx)
        input_height: Input image height
        input_width: Input image width
        opset_version: ONNX opset version (11 recommended)
        simplify: Whether to simplify ONNX graph (requires onnx-simplifier)
    """
    print(f"\n{'='*60}")
    print(f"RAFT → ONNX Export")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_file}")
    print(f"Output:     {output_file}")
    print(f"Resolution: {input_height}x{input_width}")
    print(f"Opset:      {opset_version}")
    print(f"Simplify:   {simplify}")
    print(f"{'='*60}\n")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"[DEBUG] Using device: {device}")
    if device == 'cpu':
        print(f"[DEBUG] WARNING: CUDA not available, export will be slower")

    # Load RAFT model
    print(f"\n[1/4] Loading RAFT model...")
    raft_model = load_raft_model(checkpoint_file, device)

    # Wrap model for ONNX export
    print(f"\n[2/4] Wrapping model for ONNX export...")
    model = RAFTModelWrapper(raft_model)
    model.eval()
    print(f"[DEBUG] ✓ Model wrapped")

    # Create dummy inputs: two consecutive frames
    # RAFT expects float32 images in [0, 255] range
    print(f"\n[3/4] Creating dummy inputs...")
    dummy_image1 = torch.randn(1, 3, input_height, input_width, device=device) * 255
    dummy_image2 = torch.randn(1, 3, input_height, input_width, device=device) * 255
    print(f"[DEBUG] Image1 shape: {dummy_image1.shape}, dtype: {dummy_image1.dtype}")
    print(f"[DEBUG] Image2 shape: {dummy_image2.shape}, dtype: {dummy_image2.dtype}")

    # Test forward pass
    print(f"[DEBUG] Testing forward pass...")
    try:
        with torch.no_grad():
            test_flow = model(dummy_image1, dummy_image2)
        print(f"[DEBUG] ✓ Forward pass successful")
        print(f"[DEBUG] Output flow shape: {test_flow.shape}")
    except Exception as e:
        print(f"[DEBUG] ✗ Forward pass failed: {e}")
        raise

    print(f"\n[4/4] Exporting to ONNX: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        print(f"[DEBUG] ✓ ONNX export completed")
    except Exception as e:
        print(f"[DEBUG] ✗ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    if not output_path.exists():
        print(f"[DEBUG] ✗ Output file not created!")
        sys.exit(1)

    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"[DEBUG] Output file size: {file_size:.2f} MB")
    print(f"\n✓ Exported to: {output_file}")

    # Optional: Simplify ONNX graph
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            print(f"\n[DEBUG] Simplifying ONNX graph...")
            model_onnx = onnx.load(output_file)
            print(f"[DEBUG] Original model loaded")
            model_simplified, check = onnx_simplify(model_onnx)

            if check:
                onnx.save(model_simplified, output_file)
                new_size = output_path.stat().st_size / 1024 / 1024
                print(f"[DEBUG] Simplified size: {new_size:.2f} MB (was {file_size:.2f} MB)")
                print("✓ Simplified ONNX graph")
            else:
                print("⚠ Simplification check failed, keeping original")
        except ImportError as e:
            print(f"[DEBUG] onnx-simplifier not available: {e}")
            print("⚠ onnx-simplifier not installed, skipping simplification")
            print("  Install with: pip install onnx-simplifier")
        except Exception as e:
            print(f"[DEBUG] Simplification failed: {e}")
            print(f"⚠ Simplification failed, keeping original")

    # Verify ONNX model
    print(f"\n[DEBUG] Verifying ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed")

        # Print model info
        print(f"\nModel info:")
        print(f"  Inputs: {[i.name for i in onnx_model.graph.input]}")
        print(f"  Outputs: {[o.name for o in onnx_model.graph.output]}")
        print(f"  File: {output_path.absolute()}")
    except Exception as e:
        print(f"⚠ ONNX verification failed: {e}")


def export_all_raft_models(models_dir: str = "models"):
    """Export all RAFT models to ONNX."""
    print(f"\n[DEBUG] Batch export from directory: {models_dir}")
    models = [
        # (checkpoint, output, height, width)
        ('raft-sintel.pth', 'raft-sintel_256x192.onnx', 256, 192),
        ('raft-sintel.pth', 'raft-sintel_384x256.onnx', 384, 256),
        ('raft-things.pth', 'raft-things_256x192.onnx', 256, 192),
        ('raft-small.pth', 'raft-small_256x192.onnx', 256, 192),
    ]

    models_path = Path(models_dir)
    print(f"[DEBUG] Models directory: {models_path.absolute()}")

    for checkpoint, output, height, width in models:
        checkpoint_path = models_path / checkpoint
        output_path = models_path / output

        if not checkpoint_path.exists():
            print(f"⊘ Skipping {output}: {checkpoint} not found at {checkpoint_path}")
            continue

        try:
            export_raft_onnx(str(checkpoint_path), str(output_path), height, width)
            print()
        except Exception as e:
            print(f"✗ Failed to export {output}: {e}")
            import traceback
            traceback.print_exc()
            print()


if __name__ == '__main__':
    print(f"[DEBUG] Script started")
    print(f"[DEBUG] Python: {sys.version}")
    print(f"[DEBUG] PyTorch: {torch.__version__}")
    print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[DEBUG] CUDA version: {torch.version.cuda}")

    parser = argparse.ArgumentParser(description='Export RAFT optical flow model to ONNX')
    parser.add_argument('--models-dir', default='models', help='Directory containing models')
    parser.add_argument('--checkpoint', help='RAFT checkpoint file (.pth)')
    parser.add_argument('--output', help='Output ONNX file (.onnx)')
    parser.add_argument('--height', type=int, default=256, help='Input height (default: 256)')
    parser.add_argument('--width', type=int, default=192, help='Input width (default: 192)')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--no-simplify', action='store_true', help='Skip ONNX simplification')

    args = parser.parse_args()

    print(f"[DEBUG] Arguments parsed:")
    print(f"[DEBUG]   checkpoint: {args.checkpoint}")
    print(f"[DEBUG]   output: {args.output}")
    print(f"[DEBUG]   height: {args.height}")
    print(f"[DEBUG]   width: {args.width}")
    print(f"[DEBUG]   models_dir: {args.models_dir}")

    if args.checkpoint and args.output:
        # Export single model
        print(f"[DEBUG] Mode: Single model export")
        try:
            export_raft_onnx(
                args.checkpoint,
                args.output,
                args.height,
                args.width,
                args.opset,
                not args.no_simplify
            )
            print(f"\n{'='*60}")
            print(f"SUCCESS: Export completed!")
            print(f"{'='*60}")
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"FAILED: Export failed!")
            print(f"{'='*60}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Export all models
        print(f"[DEBUG] Mode: Batch export")
        export_all_raft_models(args.models_dir)