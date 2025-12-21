# type: ignore
"""Convert RVM ONNX models to TensorRT engines."""

import tensorrt as trt
import os
import argparse
from pathlib import Path


def convert_rvm_to_tensorrt(
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
    """Convert RVM ONNX model to TensorRT engine.

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
    print(f"🔄 RVM ONNX → TensorRT Conversion")
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
    # New RVM ONNX: src (image), r1i-r4i (recurrent states with fixed shapes)
    # No downsample_ratio (hardcoded to 1.0 during export)
    profile = builder.create_optimization_profile()

    # Image input: (batch, 3, height, width) - dynamic batch dimension
    profile.set_shape("src",
                      (min_batch, 3, height, width),
                      (opt_batch, 3, height, width),
                      (max_batch, 3, height, width))

    # Recurrent states: Fixed size based on 256×192 resolution with downsample_ratio=1.0
    # These are constant shapes (no dynamic dimensions)
    # r1: [1, 16, 128, 96], r2: [1, 20, 64, 48], r3: [1, 40, 32, 24], r4: [1, 64, 16, 12]
    profile.set_shape("r1i", (1, 16, 128, 96), (1, 16, 128, 96), (1, 16, 128, 96))
    profile.set_shape("r2i", (1, 20, 64, 48), (1, 20, 64, 48), (1, 20, 64, 48))
    profile.set_shape("r3i", (1, 40, 32, 24), (1, 40, 32, 24), (1, 40, 32, 24))
    profile.set_shape("r4i", (1, 64, 16, 12), (1, 64, 16, 12), (1, 64, 16, 12))

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
    print(f" ✓")

    print(f"\n{'═'*70}")
    print(f"✅ CONVERSION COMPLETE")
    print(f"{'═'*70}")
    print(f"  Output: {output_path}")
    print(f"  Size:   {engine_size_mb:.1f} MB")
    print(f"{'═'*70}\n")

    return True


if __name__ == '__main__':
    import sys

    print(f"\n{'═'*70}")
    print(f"🚀 RVM TensorRT CONVERSION TOOL")
    print(f"{'═'*70}")
    print(f"  TensorRT: {trt.__version__}")
    print(f"{'═'*70}\n")

    parser = argparse.ArgumentParser(description='Convert RVM ONNX models to TensorRT')
    parser.add_argument('--onnx', default='models/rvm_mobilenetv3_fp16.onnx', help='Input ONNX file')
    parser.add_argument('--output', default='models/rvm_mobilenetv3_fp16_256x192_3.trt', help='Output TensorRT file')
    parser.add_argument('--height', type=int, default=256, help='Input height (default: 256)')
    parser.add_argument('--width', type=int, default=192, help='Input width (default: 192)')
    parser.add_argument('--min-batch', type=int, default=1, help='Minimum batch size')
    parser.add_argument('--opt-batch', type=int, default=3, help='Optimal batch size')
    parser.add_argument('--max-batch', type=int, default=4, help='Maximum batch size')
    parser.add_argument('--fp32', action='store_true', help='Use FP32 instead of FP16')
    parser.add_argument('--workspace', type=int, default=4, help='Workspace size in GB')

    args = parser.parse_args()

    success = convert_rvm_to_tensorrt(
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