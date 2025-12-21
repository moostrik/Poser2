# type: ignore
"""Export RTMPose PyTorch models to ONNX format."""

import torch
from mmpose.apis import init_model
import argparse
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


def export_rtmpose_onnx(config_file: str, checkpoint_file: str, output_file: str,
                        opset_version: int = 11, simplify: bool = True):
    """Export RTMPose model to ONNX format.

    Args:
        config_file: Path to model config (.py)
        checkpoint_file: Path to model checkpoint (.pth)
        output_file: Output ONNX file path (.onnx)
        opset_version: ONNX opset version (11 recommended)
        simplify: Whether to simplify ONNX graph (requires onnx-simplifier)
    """
    print(f"Loading model: {config_file}")
    model = init_model(config_file, checkpoint_file, device='cuda:0')
    model.eval()

    # Create dummy input matching expected format: uint8 BGR image [0-255]
    # Model's data_preprocessor will handle BGR->RGB and normalization
    dummy_input = torch.randint(0, 256, (1, 3, 256, 192), dtype=torch.uint8, device='cuda:0').float()

    print(f"Exporting to ONNX: {output_file}")
    print(f"  Dummy input: shape={dummy_input.shape}, dtype={dummy_input.dtype}, range=[{dummy_input.min():.1f}, {dummy_input.max():.1f}]")
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        export_params=True,
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

    print(f"✓ Exported to: {output_file}")

    # Optional: Simplify ONNX graph
    if simplify:
        try:
            import onnx
            from onnxsim import simplify

            print("Simplifying ONNX graph...")
            model_onnx = onnx.load(output_file)
            model_simplified, check = simplify(model_onnx)

            if check:
                onnx.save(model_simplified, output_file)
                print("✓ Simplified ONNX graph")
            else:
                print("⚠ Simplification check failed, keeping original")
        except ImportError:
            print("⚠ onnx-simplifier not installed, skipping simplification")
            print("  Install with: pip install onnx-simplifier")

    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed")
    except Exception as e:
        print(f"⚠ ONNX verification failed: {e}")


def export_all_models(models_dir: str = "models"):
    """Export all RTMPose models to ONNX."""
    models = [
        ('rtmpose-l_8xb256-420e_aic-coco-256x192.py',
         'rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.pth',
         'rtmpose-l_256x192.onnx'),

        ('rtmpose-m_8xb256-420e_aic-coco-256x192.py',
         'rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth',
         'rtmpose-m_256x192.onnx'),

        ('rtmpose-s_8xb256-420e_aic-coco-256x192.py',
         'rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth',
         'rtmpose-s_256x192.onnx'),

        ('rtmpose-t_8xb256-420e_aic-coco-256x192.py',
         'rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth',
         'rtmpose-t_256x192.onnx'),

        # ('wb_rtmpose-l_8xb64-270e_coco-wholebody-256x192.py',
        #  'wb_rtmpose-l_simcc-ucoco_dw-ucoco_270e-256x192-4d6dfc62_20230728.pth',
        #  'wb_rtmpose-l_256x192.onnx'),

        # ('wb_rtmpose-m_8xb64-270e_coco-wholebody-256x192.py',
        #  'wb_rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.pth',
        #  'wb_rtmpose-m_256x192.onnx'),

        # ('wb_rtmpose-s_8xb64-270e_coco-wholebody-256x192.py',
        #  'wb_rtmpose-s_simcc-ucoco_dw-ucoco_270e-256x192-3fd922c8_20230728.pth',
        #  'wb_rtmpose-s_256x192.onnx'),
    ]

    models_path = Path(models_dir)

    for config, checkpoint, output in models:
        config_path = models_path / config
        checkpoint_path = models_path / checkpoint
        output_path = models_path / output

        if not config_path.exists() or not checkpoint_path.exists():
            print(f"⊘ Skipping {output}: missing files")
            continue

        try:
            export_rtmpose_onnx(str(config_path), str(checkpoint_path), str(output_path))
            print()
        except Exception as e:
            print(f"✗ Failed to export {output}: {e}")
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export RTMPose models to ONNX')
    parser.add_argument('--models-dir', default='models', help='Directory containing models')
    parser.add_argument('--config', help='Single config file to export')
    parser.add_argument('--checkpoint', help='Single checkpoint file')
    parser.add_argument('--output', help='Output ONNX file')

    args = parser.parse_args()

    if args.config and args.checkpoint and args.output:
        # Export single model
        export_rtmpose_onnx(args.config, args.checkpoint, args.output)
    else:
        # Export all models
        export_all_models(args.models_dir)