# type: ignore
"""Convert RAFT ONNX models to TensorRT engines."""

import tensorrt as trt
import os
import argparse
from pathlib import Path


def convert_raft_to_tensorrt(
    onnx_path: str,
    output_path: str,
    height: int,
    width: int,
    min_batch: int = 1,
    opt_batch: int = 3,
    max_batch: int = 4,
    fp16: bool = True,
    workspace_gb: int = 4
):
    """Convert RAFT ONNX model to TensorRT engine.

    Args:
        onnx_path: Path to input ONNX file
        output_path: Path to output TensorRT engine
        height: Input image height
        width: Input image width
        min_batch: Minimum batch size
        opt_batch: Optimal batch size (used for optimization)
        max_batch: Maximum batch size
        fp16: Enable FP16 precision
        workspace_gb: Workspace size in GB
    """
    print(f"\n{'═'*70}")
    print(f"🔄 RAFT ONNX → TensorRT Conversion")
    print(f"{'═'*70}")
    print(f"  Input:      {onnx_path}")
    print(f"  Output:     {output_path}")
    print(f"  Resolution: {height}×{width}")
    print(f"  Batch:      min={min_batch}, opt={opt_batch}, max={max_batch}")
    print(f"  Precision:  {'FP16' if fp16 else 'FP32'}")
    print(f"  Workspace:  {workspace_gb} GB")
    print(f"{'═'*70}\n")

    # Check if ONNX file exists
    if not os.path.exists(onnx_path):
        print(f"❌ ERROR: ONNX file not found: {onnx_path}")
        return False

    print(f"✓ Found ONNX file ({os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB)")

    # Create TensorRT builder
    print("📦 Creating TensorRT builder...", end='', flush=True)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    print(" ✓")

    # Parse ONNX model
    print("📖 Parsing ONNX model...", end='', flush=True)
    with open(onnx_path, "rb") as f:
        model_data = f.read()
        success = parser.parse(model_data)

    if not success:
        print(" ✗")
        print("\n❌ Failed to parse ONNX file!")
        for i in range(parser.num_errors):
            print(f"   Error {i}: {parser.get_error(i)}")
        return False

    print(f" ✓ ({network.num_inputs} inputs, {network.num_outputs} outputs)")

    # Print input/output info
    print("\n📋 Network Information:")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  Input {i}:  {inp.name} {inp.shape}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  Output {i}: {out.name} {out.shape}")

    # Configure builder
    print("\n⚙️  Configuring builder...")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    print(f"  ├─ Workspace: {workspace_gb} GB")

    # Create optimization profile for dynamic batch size
    # RAFT has 2 inputs: image1 and image2, both (batch, 3, height, width)
    profile = builder.create_optimization_profile()

    # Set shapes for both inputs
    profile.set_shape("image1",
                      (min_batch, 3, height, width),
                      (opt_batch, 3, height, width),
                      (max_batch, 3, height, width))
    profile.set_shape("image2",
                      (min_batch, 3, height, width),
                      (opt_batch, 3, height, width),
                      (max_batch, 3, height, width))

    config.add_optimization_profile(profile)
    print(f"  ├─ Optimization profile: batch {min_batch}/{opt_batch}/{max_batch}, shape {height}×{width}")

    # Enable FP16 if requested
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print(f"  └─ FP16 enabled ✓")
    else:
        print(f"  └─ Using FP32")

    # Build engine
    print("\n🔨 Building TensorRT engine...")
    print("   ⏱️  This may take 2-5 minutes, please wait...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("\n❌ Failed to build TensorRT engine!")
        return False

    # Convert IHostMemory to bytes
    engine_bytes = bytes(serialized_engine)
    engine_size_mb = len(engine_bytes) / 1024 / 1024
    print(f"   ✓ Engine built ({engine_size_mb:.1f} MB)")

    # Save engine
    print(f"\n💾 Saving engine...", end='', flush=True)
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(engine_bytes)
    print(f"{'═'*70}")
    print(f"  Output: {output_path}")
    print(f"  Size:   {engine_size_mb:.1f} MB")
    print(f"{'═'*70}\n")

    return True


def convert_all_raft_models(
    models_dir: str = "models",
    min_batch: int = 1,
    opt_batch: int = 3,
    max_batch: int = 4
):
    """Convert all RAFT ONNX models to TensorRT."""
    print(f"\n{'═'*70}")
    print(f"📦 BATCH TENSORRT CONVERSION")
    print(f"{'═'*70}")

    models = [
        # (onnx_file, trt_file, height, width)
        ('raft-sintel_256x192.onnx', f'raft-sintel_256x192_{opt_batch}.trt', 256, 192),
        ('raft-small_256x192.onnx', f'raft-small_256x192_{opt_batch}.trt', 256, 192),
        # ('raft-sintel_384x288.onnx', f'raft-sintel_384x288_{opt_batch}.trt', 384, 288),
        # ('raft-sintel_512x384.onnx', f'raft-sintel_512x384_{opt_batch}.trt', 512, 384),
    ]

    models_path = Path(models_dir)
    print(f"  Models directory: {models_path.absolute()}")
    print(f"  Total models:     {len(models)}")
    print(f"  Batch config:     min={min_batch}, opt={opt_batch}, max={max_batch}")
    print(f"{'═'*70}")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for idx, (onnx_file, trt_file, height, width) in enumerate(models, 1):
        onnx_path = models_path / onnx_file
        trt_path = models_path / trt_file

        print(f"\n{'─'*70}")
        print(f"[{idx}/{len(models)}] {onnx_file} → {trt_file}")
        print(f"{'─'*70}")

        if not onnx_path.exists():
            print(f"⊘ Skipping: {onnx_file} not found")
            skip_count += 1
            continue

        try:
            success = convert_raft_to_tensorrt(
                str(onnx_path),
                str(trt_path),
                height,
                width,
                min_batch,
                opt_batch,
                max_batch
            )
            if success:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            fail_count += 1

    # Summary
    print(f"\n{'═'*70}")
    print(f"📊 CONVERSION SUMMARY")
    print(f"{'═'*70}")
    print(f"  ✅ Successful: {success_count}")
    print(f"  ⊘  Skipped:    {skip_count}")
    print(f"  ❌ Failed:     {fail_count}")
    print(f"  ━  Total:      {len(models)}")
    print(f"{'═'*70}\n")


if __name__ == '__main__':
    import sys

    print(f"\n{'═'*70}")
    print(f"🚀 RAFT TensorRT CONVERSION TOOL")
    print(f"{'═'*70}")
    print(f"  TensorRT: {trt.__version__}")
    print(f"{'═'*70}\n")

    parser = argparse.ArgumentParser(description='Convert RAFT ONNX models to TensorRT')
    parser.add_argument('--models-dir', default='models', help='Directory containing models')
    parser.add_argument('--onnx', help='Input ONNX file')
    parser.add_argument('--output', help='Output TensorRT file')
    parser.add_argument('--height', type=int, default=256, help='Input height')
    parser.add_argument('--width', type=int, default=192, help='Input width')
    parser.add_argument('--min-batch', type=int, default=1, help='Minimum batch size')
    parser.add_argument('--opt-batch', type=int, default=3, help='Optimal batch size')
    parser.add_argument('--max-batch', type=int, default=4, help='Maximum batch size')
    parser.add_argument('--fp32', action='store_true', help='Use FP32 instead of FP16')
    parser.add_argument('--workspace', type=int, default=4, help='Workspace size in GB')

    args = parser.parse_args()

    if args.onnx and args.output:
        # Convert single model
        print("Mode: Single model conversion\n")
        success = convert_raft_to_tensorrt(
            args.onnx,
            args.output,
            args.height,
            args.width,
            args.min_batch,
            args.opt_batch,
            args.max_batch,
            not args.fp32,
            args.workspace
        )
        sys.exit(0 if success else 1)
    else:
        # Convert all models
        print("Mode: Batch conversion\n")
        convert_all_raft_models(args.models_dir, args.min_batch, args.opt_batch, args.max_batch)
