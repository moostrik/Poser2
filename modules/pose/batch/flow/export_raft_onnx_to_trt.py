# type: ignore
"""Convert RAFT ONNX models to TensorRT engines.

Usage:
    # Convert RAFT Sintel 256x192 (auto-detects dimensions and precision)
    python modules/pose/batch/flow/export_raft_onnx_to_trt.py --onnx models/raft-sintel_256x192_i12.onnx --output models/raft-sintel_256x192_i12_b3.trt

    # Convert 384x288 model
    python modules/pose/batch/flow/export_raft_onnx_to_trt.py --onnx models/raft-sintel_384x288_i12.onnx --output models/raft-sintel_384x288_i12_b3.trt

    # Convert 512x384 model
    python modules/pose/batch/flow/export_raft_onnx_to_trt.py --onnx models/raft-sintel_512x384_i12.onnx --output models/raft-sintel_512x384_i12_b3.trt
"""

import tensorrt as trt
import os
import argparse
from pathlib import Path


def convert_raft_to_tensorrt(
    onnx_path: str,
    output_path: str | None = None,
    min_batch: int = 1,
    opt_batch: int = 3,
    max_batch: int = 4,
    workspace_gb: int = 8,
    fp16_override: bool = False
) -> bool:
    """Convert RAFT ONNX model to TensorRT engine.

    Args:
        onnx_path: Path to input ONNX file
        output_path: Path to output TensorRT engine (auto-generated from input if not provided)
        min_batch: Minimum batch size
        opt_batch: Optimal batch size (used for optimization)
        max_batch: Maximum batch size (set to opt_batch if never exceeded)
        workspace_gb: Workspace size in GB
        fp16_override: Force FP16 precision even if ONNX is FP32 (default: False)

    Returns:
        bool: True if successful, False otherwise
    """
    # Auto-generate output path from input if not provided
    if output_path is None:
        onnx_path_obj = Path(onnx_path)
        output_path = str(onnx_path_obj.with_suffix('.trt'))
    print(f"\n{'═'*70}")
    print(f"🔄 RAFT ONNX → TensorRT Conversion")
    print(f"{'═'*70}")
    print(f"  Input:      {onnx_path}")
    print(f"  Output:     {output_path}")
    print(f"  Batch:      min={min_batch}, opt={opt_batch}, max={max_batch}")
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

    # Auto-detect dimensions from ONNX model
    print("\n🔍 Auto-detecting dimensions from ONNX model...")
    height = None
    width = None
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        if inp.name == "image1":  # RAFT uses 'image1' and 'image2' for inputs
            shape = inp.shape
            # Shape is (batch, channels, height, width)
            if len(shape) >= 4:
                height = shape[2] if shape[2] != -1 else None
                width = shape[3] if shape[3] != -1 else None
                if height and width:
                    print(f"  ✓ Detected resolution: {height}×{width}")
                break

    # Validate dimensions were found
    if height is None or width is None:
        print("\n❌ ERROR: Could not auto-detect dimensions from ONNX model!")
        print("   The 'image1' input tensor must have a fixed shape [batch, 3, height, width].")
        return False

    # Print input/output info
    print("\n📋 Network Information:")
    print(f"  Resolution: {height}×{width}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  Input {i}:  {inp.name} {inp.shape}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  Output {i}: {out.name} {out.shape}")

    # Detect ONNX model precision
    print("\n🔍 Detecting ONNX Model Precision:")
    import onnx
    onnx_model = onnx.load(onnx_path)

    # Check initializer (weights) precision
    use_fp16 = False
    detected_precision = "UNKNOWN"
    if onnx_model.graph.initializer:
        onnx_dtype_map = {1: "FLOAT32", 10: "FLOAT16", 2: "UINT8", 3: "INT8", 6: "INT32", 7: "INT64"}
        weight_dtypes = set()
        for initializer in onnx_model.graph.initializer:
            dtype_str = onnx_dtype_map.get(initializer.data_type, f"Unknown({initializer.data_type})")
            weight_dtypes.add(dtype_str)

        # Determine precision (prefer FP16 if found, otherwise FP32)
        if "FLOAT16" in weight_dtypes:
            detected_precision = "FP16"
            use_fp16 = True
        elif "FLOAT32" in weight_dtypes:
            detected_precision = "FP32"
            use_fp16 = False
        else:
            detected_precision = ", ".join(sorted(weight_dtypes))

        print(f"  ONNX weights: {', '.join(sorted(weight_dtypes))}")
        print(f"  ✓ Will build TensorRT engine with: {detected_precision}")
    else:
        print("  ⚠️  ONNX weights: No initializers found")
        print("  Defaulting to FP32")
        detected_precision = "FP32"
        use_fp16 = False

    # Apply FP16 override if requested
    if fp16_override:
        use_fp16 = True
        build_precision = "FP16 (forced override)"
        print(f"  ⚠️  FP16 override: Forcing FP16 build from {detected_precision} ONNX")
    else:
        build_precision = f"{detected_precision} (auto-detected)"

    # Show build configuration
    print("\n🎯 TensorRT Build Configuration:")
    print(f"  Precision: {build_precision}")

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

    # Set precision mode based on detected ONNX precision
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print(f"  └─ Precision: FP16")
    else:
        print(f"  └─ Precision: FP32")

    # Build engine
    print(f"\n🔨 Building TensorRT engine ({detected_precision})...)")
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

    # Verify built engine precision (same method as TRTDetection._setup)
    print("🔍 Verifying built precision...", end='', flush=True)
    runtime_verify = trt.Runtime(logger)
    with open(output_path, 'rb') as f:
        built_engine = runtime_verify.deserialize_cuda_engine(f.read())

    # Get input tensor dtype
    input_name = built_engine.get_tensor_name(0)
    input_dtype = built_engine.get_tensor_dtype(input_name)

    precision_map = {
        trt.DataType.FLOAT: "FP32",
        trt.DataType.HALF: "FP16",
        trt.DataType.INT8: "INT8",
    }
    actual_precision = precision_map.get(input_dtype, "UNKNOWN")

    if fp16_override and actual_precision == "FP16":
        print(f" ✓ (Built as {actual_precision} ✅)")
    elif fp16_override and actual_precision != "FP16":
        print(f" ⚠️ (Requested FP16 but built as {actual_precision})")
    else:
        print(f" ✓ (Built as {actual_precision})")

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
    print(f"🚀 RAFT TensorRT CONVERSION TOOL")
    print(f"{'═'*70}")
    print(f"  TensorRT: {trt.__version__}")
    print(f"{'═'*70}\n")

    parser = argparse.ArgumentParser(
        description='Convert RAFT ONNX model to TensorRT engine',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument('--onnx', required=True,
                        help='Path to input ONNX file')
    parser.add_argument('--output', default=None,
                        help='Path to output TensorRT engine file (default: replace .onnx with .trt)')

    # Optional arguments with defaults
    parser.add_argument('--min-batch', type=int, default=1,
                        help='Minimum batch size (default: 1)')
    parser.add_argument('--opt-batch', type=int, default=3,
                        help='Optimal batch size for optimization (default: 3)')
    parser.add_argument('--max-batch', type=int, default=3,
                        help='Maximum batch size (default: 3, set to opt-batch if never exceeded)')
    parser.add_argument('--workspace', type=int, default=8,
                        help='Workspace size in GB (default: 8)')
    parser.add_argument('--fp16', action='store_true',
                        help='Force FP16 precision even if ONNX is FP32 (faster inference, minimal accuracy loss)')

    args = parser.parse_args()

    success = convert_raft_to_tensorrt(
        args.onnx,
        args.output,
        args.min_batch,
        args.opt_batch,
        args.max_batch,
        args.workspace,
        args.fp16
    )

    sys.exit(0 if success else 1)
