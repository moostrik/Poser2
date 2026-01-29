# type: ignore
"""Diagnostic script to compare model precision behavior.

Compares outputs between any two models (ONNX vs ONNX, TRT vs TRT, or ONNX vs TRT)
to identify precision-related divergence in RTMPose SimCC outputs.

Usage:
    # Compare ONNX vs TRT
    python modules/pose/batch/detection/diagnostic_precision.py --model1 models/rtmpose-l_256x192_fp16.onnx --model2 models/rtmpose-l_256x192_b3_fp16.trt

    # Compare two ONNX models (e.g., FP32 vs FP16)
    python modules/pose/batch/detection/diagnostic_precision.py --model1 models/rtmpose-l_256x192_fp32.onnx --model2 models/rtmpose-l_256x192_fp16.onnx

    # Compare two TRT engines
    python modules/pose/batch/detection/diagnostic_precision.py --model1 models/rtmpose-l_256x192_b3_fp32.trt --model2 models/rtmpose-l_256x192_b3_fp16.trt

    # Legacy syntax (still supported)
    python modules/pose/batch/detection/diagnostic_precision.py --onnx models/rtmpose-l_256x192.onnx --trt models/rtmpose-l_256x192_b3.trt --batch 3
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import onnxruntime as ort
import tensorrt as trt


# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def get_onnx_precision(model_path: str) -> Tuple[str, np.dtype]:
    """Detect ONNX model precision from input tensor type."""
    session = ort.InferenceSession(
        model_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    input_info = session.get_inputs()[0]
    if 'float16' in input_info.type.lower():
        return "FP16", np.float16
    else:
        return "FP32", np.float32


def get_trt_precision(engine_path: str) -> Tuple[str, np.dtype]:
    """Detect TRT engine precision from input tensor type."""
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")
    
    # Find input tensor
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            dtype = engine.get_tensor_dtype(name)
            if dtype == trt.DataType.HALF:
                return "FP16", np.float16
            else:
                return "FP32", np.float32
    
    return "FP32", np.float32


def get_model_type(model_path: str) -> str:
    """Detect model type from file extension."""
    path = Path(model_path)
    ext = path.suffix.lower()
    if ext == '.onnx':
        return 'onnx'
    elif ext == '.trt' or ext == '.engine':
        return 'trt'
    else:
        raise ValueError(f"Unknown model type for file: {model_path}")


def get_model_precision(model_path: str) -> Tuple[str, np.dtype]:
    """Detect model precision based on model type."""
    model_type = get_model_type(model_path)
    if model_type == 'onnx':
        return get_onnx_precision(model_path)
    else:
        return get_trt_precision(model_path)


def get_model_input_shape(model_path: str) -> Tuple[int, int, int, int]:
    """Get input shape from model (batch, channels, height, width)."""
    model_type = get_model_type(model_path)
    
    if model_type == 'onnx':
        session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        input_info = session.get_inputs()[0]
        shape = input_info.shape
        # Handle dynamic dimensions
        batch = shape[0] if isinstance(shape[0], int) else -1
        channels = shape[1] if isinstance(shape[1], int) else 3
        height = shape[2] if isinstance(shape[2], int) else 256
        width = shape[3] if isinstance(shape[3], int) else 192
        return batch, channels, height, width
    else:
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(model_path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                shape = engine.get_tensor_shape(name)
                batch = shape[0] if shape[0] != -1 else -1
                channels = shape[1] if len(shape) > 1 else 3
                height = shape[2] if len(shape) > 2 else 256
                width = shape[3] if len(shape) > 3 else 192
                return batch, channels, height, width
        
        return -1, 3, 256, 192


def run_inference(model_path: str, input_data: np.ndarray) -> Dict[str, np.ndarray]:
    """Run inference on any supported model type, converting input dtype as needed."""
    model_type = get_model_type(model_path)
    
    # Get the model's expected dtype and convert input if needed
    _, expected_dtype = get_model_precision(model_path)
    input_converted = input_data.astype(expected_dtype)
    
    if model_type == 'onnx':
        return run_onnx_inference(model_path, input_converted)
    else:
        return run_trt_inference(model_path, input_converted)


def run_onnx_inference(model_path: str, input_data: np.ndarray) -> Dict[str, np.ndarray]:
    """Run ONNX Runtime inference and capture all outputs."""
    
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3  # Suppress verbose logging
    
    session = ort.InferenceSession(
        model_path, sess_options,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    
    outputs = session.run(output_names, {input_name: input_data})
    
    return {name: out for name, out in zip(output_names, outputs)}


def run_trt_inference(engine_path: str, input_data: np.ndarray) -> Dict[str, np.ndarray]:
    """Run TensorRT inference and capture outputs."""
    
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    if engine is None:
        raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")
    
    context = engine.create_execution_context()
    
    # Get tensor info and allocate buffers
    outputs = {}
    input_name = None
    output_tensors = {}
    
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = list(engine.get_tensor_shape(name))
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        
        # Handle dynamic batch dimension
        if shape[0] == -1:
            shape[0] = input_data.shape[0]
        
        # Determine numpy dtype
        if dtype == trt.DataType.HALF:
            np_dtype = np.float16
        elif dtype == trt.DataType.FLOAT:
            np_dtype = np.float32
        else:
            np_dtype = np.float32
        
        if mode == trt.TensorIOMode.INPUT:
            input_name = name
            # Convert input to correct dtype
            buffer = torch.from_numpy(input_data.astype(np_dtype)).cuda().contiguous()
            context.set_tensor_address(name, buffer.data_ptr())
            context.set_input_shape(name, tuple(input_data.shape))
        else:
            # Output tensor
            buffer = torch.empty(shape, dtype=torch.float16 if np_dtype == np.float16 else torch.float32, device='cuda')
            output_tensors[name] = buffer
            context.set_tensor_address(name, buffer.data_ptr())
    
    # Execute inference
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        success = context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    
    if not success:
        raise RuntimeError("TensorRT inference failed")
    
    # Copy outputs to CPU
    for name, tensor in output_tensors.items():
        outputs[name] = tensor.cpu().numpy()
    
    return outputs


def decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray, split_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Decode SimCC format to keypoints and scores."""
    # simcc_x: (B, K, W*2), simcc_y: (B, K, H*2)
    
    # Get argmax positions
    x_locs = np.argmax(simcc_x, axis=-1)  # (B, K)
    y_locs = np.argmax(simcc_y, axis=-1)  # (B, K)
    
    # Get max values for confidence
    x_max = np.max(simcc_x, axis=-1)
    y_max = np.max(simcc_y, axis=-1)
    
    # Combine scores (geometric mean approximation)
    scores = np.sqrt(np.maximum(x_max, 0) * np.maximum(y_max, 0))
    
    # Scale back to pixel coordinates
    x_coords = x_locs / split_ratio
    y_coords = y_locs / split_ratio
    
    # Stack to (B, K, 2)
    keypoints = np.stack([x_coords, y_coords], axis=-1)
    
    return keypoints, scores


def check_numerical_stability(arr: np.ndarray, name: str) -> Dict:
    """Check array for numerical issues common with FP16."""
    arr_f32 = arr.astype(np.float32)
    
    stats = {
        'name': name,
        'shape': arr.shape,
        'dtype': str(arr.dtype),
        'min': float(arr_f32.min()),
        'max': float(arr_f32.max()),
        'mean': float(arr_f32.mean()),
        'std': float(arr_f32.std()),
        'num_nan': int(np.sum(np.isnan(arr_f32))),
        'num_inf': int(np.sum(np.isinf(arr_f32))),
        'num_zero': int(np.sum(arr_f32 == 0)),
    }
    
    # Check for denormal values (FP16 underflow risk)
    fp16_min_normal = 6.1e-5
    stats['num_denormal'] = int(np.sum((np.abs(arr_f32) < fp16_min_normal) & (arr_f32 != 0)))
    
    return stats


def compare_simcc_outputs(
    outputs1: Dict[str, np.ndarray],
    outputs2: Dict[str, np.ndarray],
    name1: str = "Model1",
    name2: str = "Model2",
    split_ratio: float = 2.0
) -> None:
    """Compare SimCC outputs between two models."""
    
    print(f"\n{'='*80}")
    print(f"SIMCC OUTPUT COMPARISON: {name1} vs {name2}")
    print(f"{'='*80}")
    
    # Get simcc outputs
    m1_x = outputs1.get('simcc_x')
    m1_y = outputs1.get('simcc_y')
    m2_x = outputs2.get('simcc_x')
    m2_y = outputs2.get('simcc_y')
    
    if m1_x is None or m2_x is None:
        print("❌ Could not find simcc_x in outputs")
        print(f"   {name1} outputs: {list(outputs1.keys())}")
        print(f"   {name2} outputs: {list(outputs2.keys())}")
        return
    
    # Convert to float32 for comparison
    m1_x_f32 = m1_x.astype(np.float32)
    m1_y_f32 = m1_y.astype(np.float32)
    m2_x_f32 = m2_x.astype(np.float32)
    m2_y_f32 = m2_y.astype(np.float32)
    
    # === SimCC X Analysis ===
    print(f"\n{'─'*40}")
    print("📊 simcc_x Analysis")
    print(f"{'─'*40}")
    
    print(f"\n  {name1} simcc_x:")
    m1_x_stats = check_numerical_stability(m1_x, f"{name1}_simcc_x")
    print(f"    Shape: {m1_x_stats['shape']}, Dtype: {m1_x_stats['dtype']}")
    print(f"    Range: [{m1_x_stats['min']:.6f}, {m1_x_stats['max']:.6f}]")
    print(f"    Mean: {m1_x_stats['mean']:.6f}, Std: {m1_x_stats['std']:.6f}")
    print(f"    Sum (first keypoint): {m1_x_f32[0, 0].sum():.6f}")
    if m1_x_stats['num_nan'] > 0:
        print(f"    ⚠️  NaN values: {m1_x_stats['num_nan']}")
    if m1_x_stats['num_denormal'] > 0:
        print(f"    ⚠️  Denormal values: {m1_x_stats['num_denormal']}")
    
    print(f"\n  {name2} simcc_x:")
    m2_x_stats = check_numerical_stability(m2_x, f"{name2}_simcc_x")
    print(f"    Shape: {m2_x_stats['shape']}, Dtype: {m2_x_stats['dtype']}")
    print(f"    Range: [{m2_x_stats['min']:.6f}, {m2_x_stats['max']:.6f}]")
    print(f"    Mean: {m2_x_stats['mean']:.6f}, Std: {m2_x_stats['std']:.6f}")
    print(f"    Sum (first keypoint): {m2_x_f32[0, 0].sum():.6f}")
    if m2_x_stats['num_nan'] > 0:
        print(f"    ⚠️  NaN values: {m2_x_stats['num_nan']}")
    if m2_x_stats['num_denormal'] > 0:
        print(f"    ⚠️  Denormal values: {m2_x_stats['num_denormal']}")
    
    # Differences
    x_abs_diff = np.abs(m1_x_f32 - m2_x_f32)
    x_max_diff = np.max(x_abs_diff)
    x_mean_diff = np.mean(x_abs_diff)
    
    print(f"\n  simcc_x Difference:")
    print(f"    Max absolute diff: {x_max_diff:.6e}")
    print(f"    Mean absolute diff: {x_mean_diff:.6e}")
    
    # Argmax comparison
    m1_x_argmax = np.argmax(m1_x_f32, axis=-1)
    m2_x_argmax = np.argmax(m2_x_f32, axis=-1)
    x_argmax_match = np.mean(m1_x_argmax == m2_x_argmax)
    print(f"    Argmax match rate: {x_argmax_match:.2%}")
    
    if x_argmax_match < 1.0:
        # Show which keypoints differ
        batch_size, num_keypoints = m1_x_argmax.shape
        mismatches = []
        for b in range(batch_size):
            for k in range(num_keypoints):
                if m1_x_argmax[b, k] != m2_x_argmax[b, k]:
                    mismatches.append(f"[B{b},K{k}]: {name1}={m1_x_argmax[b,k]}, {name2}={m2_x_argmax[b,k]}")
        if len(mismatches) <= 10:
            print(f"    Mismatches: {mismatches}")
        else:
            print(f"    First 10 mismatches: {mismatches[:10]}")
    
    # === SimCC Y Analysis ===
    print(f"\n{'─'*40}")
    print("📊 simcc_y Analysis")
    print(f"{'─'*40}")
    
    print(f"\n  {name1} simcc_y:")
    m1_y_stats = check_numerical_stability(m1_y, f"{name1}_simcc_y")
    print(f"    Shape: {m1_y_stats['shape']}, Dtype: {m1_y_stats['dtype']}")
    print(f"    Range: [{m1_y_stats['min']:.6f}, {m1_y_stats['max']:.6f}]")
    print(f"    Mean: {m1_y_stats['mean']:.6f}, Std: {m1_y_stats['std']:.6f}")
    print(f"    Sum (first keypoint): {m1_y_f32[0, 0].sum():.6f}")
    if m1_y_stats['num_nan'] > 0:
        print(f"    ⚠️  NaN values: {m1_y_stats['num_nan']}")
    if m1_y_stats['num_denormal'] > 0:
        print(f"    ⚠️  Denormal values: {m1_y_stats['num_denormal']}")
    
    print(f"\n  {name2} simcc_y:")
    m2_y_stats = check_numerical_stability(m2_y, f"{name2}_simcc_y")
    print(f"    Shape: {m2_y_stats['shape']}, Dtype: {m2_y_stats['dtype']}")
    print(f"    Range: [{m2_y_stats['min']:.6f}, {m2_y_stats['max']:.6f}]")
    print(f"    Mean: {m2_y_stats['mean']:.6f}, Std: {m2_y_stats['std']:.6f}")
    print(f"    Sum (first keypoint): {m2_y_f32[0, 0].sum():.6f}")
    if m2_y_stats['num_nan'] > 0:
        print(f"    ⚠️  NaN values: {m2_y_stats['num_nan']}")
    if m2_y_stats['num_denormal'] > 0:
        print(f"    ⚠️  Denormal values: {m2_y_stats['num_denormal']}")
    
    # Differences
    y_abs_diff = np.abs(m1_y_f32 - m2_y_f32)
    y_max_diff = np.max(y_abs_diff)
    y_mean_diff = np.mean(y_abs_diff)
    
    print(f"\n  simcc_y Difference:")
    print(f"    Max absolute diff: {y_max_diff:.6e}")
    print(f"    Mean absolute diff: {y_mean_diff:.6e}")
    
    # Argmax comparison
    m1_y_argmax = np.argmax(m1_y_f32, axis=-1)
    m2_y_argmax = np.argmax(m2_y_f32, axis=-1)
    y_argmax_match = np.mean(m1_y_argmax == m2_y_argmax)
    print(f"    Argmax match rate: {y_argmax_match:.2%}")
    
    if y_argmax_match < 1.0:
        batch_size, num_keypoints = m1_y_argmax.shape
        mismatches = []
        for b in range(batch_size):
            for k in range(num_keypoints):
                if m1_y_argmax[b, k] != m2_y_argmax[b, k]:
                    mismatches.append(f"[B{b},K{k}]: {name1}={m1_y_argmax[b,k]}, {name2}={m2_y_argmax[b,k]}")
        if len(mismatches) <= 10:
            print(f"    Mismatches: {mismatches}")
        else:
            print(f"    First 10 mismatches: {mismatches[:10]}")
    
    # === Decoded Keypoints Comparison ===
    print(f"\n{'─'*40}")
    print("📊 Decoded Keypoints Comparison")
    print(f"{'─'*40}")
    
    m1_kpts, m1_scores = decode_simcc(m1_x_f32, m1_y_f32, split_ratio)
    m2_kpts, m2_scores = decode_simcc(m2_x_f32, m2_y_f32, split_ratio)
    
    kpt_diff = np.abs(m1_kpts - m2_kpts)
    score_diff = np.abs(m1_scores - m2_scores)
    
    print(f"\n  Keypoint coordinate differences:")
    print(f"    Max diff: {np.max(kpt_diff):.4f} pixels")
    print(f"    Mean diff: {np.mean(kpt_diff):.4f} pixels")
    print(f"    Keypoints within 1px: {np.mean(np.max(kpt_diff, axis=-1) < 1.0):.2%}")
    print(f"    Keypoints within 5px: {np.mean(np.max(kpt_diff, axis=-1) < 5.0):.2%}")
    
    print(f"\n  Score differences:")
    print(f"    {name1} scores range: [{m1_scores.min():.4f}, {m1_scores.max():.4f}]")
    print(f"    {name2} scores range: [{m2_scores.min():.4f}, {m2_scores.max():.4f}]")
    print(f"    Max diff: {np.max(score_diff):.6f}")
    print(f"    Mean diff: {np.mean(score_diff):.6f}")
    
    # === First 10 values comparison ===
    print(f"\n{'─'*40}")
    print("📊 First 10 Values Comparison (simcc_x[0,0,:10])")
    print(f"{'─'*40}")
    print(f"  {name1}: {m1_x_f32[0, 0, :10]}")
    print(f"  {name2}: {m2_x_f32[0, 0, :10]}")
    print(f"  Diff: {x_abs_diff[0, 0, :10]}")
    
    # === Summary ===
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    total_argmax_match = (x_argmax_match + y_argmax_match) / 2
    
    if total_argmax_match >= 0.99:
        print(f"✅ {name1} and {name2} outputs are nearly identical")
    elif total_argmax_match >= 0.90:
        print("⚠️  Minor differences detected - may cause some keypoint errors")
    elif total_argmax_match >= 0.50:
        print("❌ Significant differences detected - likely precision issue")
    else:
        print("❌ MAJOR differences detected - outputs are fundamentally different")
        print("   This suggests models have different precision or configuration")
    
    print(f"\n  Overall argmax match rate: {total_argmax_match:.2%}")
    print(f"  simcc_x argmax match: {x_argmax_match:.2%}")
    print(f"  simcc_y argmax match: {y_argmax_match:.2%}")
    
    if total_argmax_match < 0.99:
        print("\n📋 Recommended actions:")
        if x_max_diff > 1.0 or y_max_diff > 1.0:
            print("   1. Force SimCC output layers to FP32 in TRT conversion")
            print("   2. Use --mixed-precision flag when rebuilding TRT engine")
        if m1_x_stats['mean'] * m2_x_stats['mean'] < 0:  # Different signs
            print("   3. Check if output tensor names are swapped (simcc_x vs simcc_y)")
        print("   4. Try building TRT from FP32 ONNX with FP16 flag (auto mixed-precision)")


def main():
    print(f"\n{'='*80}")
    print("🔬 RTMPose Model Precision DIAGNOSTIC")
    print(f"{'='*80}")
    
    parser = argparse.ArgumentParser(
        description='Compare outputs between two models (ONNX and/or TRT) for RTMPose',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # New flexible interface
    parser.add_argument('--model1', default=None,
                        help='Path to first model file (ONNX or TRT)')
    parser.add_argument('--model2', default=None,
                        help='Path to second model file (ONNX or TRT)')
    
    # Legacy interface (still supported)
    parser.add_argument('--onnx', default=None,
                        help='(Legacy) Path to ONNX model file')
    parser.add_argument('--trt', default=None,
                        help='(Legacy) Path to TensorRT engine file')
    
    parser.add_argument('--batch', type=int, default=1,
                        help='Batch size for test inference (default: 1)')
    parser.add_argument('--height', type=int, default=None,
                        help='Input height (auto-detected if not specified)')
    parser.add_argument('--width', type=int, default=None,
                        help='Input width (auto-detected if not specified)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Handle legacy and new interface
    if args.model1 and args.model2:
        model1_path = args.model1
        model2_path = args.model2
    elif args.onnx and args.trt:
        model1_path = args.onnx
        model2_path = args.trt
    else:
        print("❌ ERROR: Please provide either --model1 and --model2, or --onnx and --trt")
        parser.print_help()
        return 1
    
    # Validate paths
    if not Path(model1_path).exists():
        print(f"❌ Model 1 not found: {model1_path}")
        return 1
    if not Path(model2_path).exists():
        print(f"❌ Model 2 not found: {model2_path}")
        return 1
    
    # Detect model types
    model1_type = get_model_type(model1_path)
    model2_type = get_model_type(model2_path)
    model1_name = f"{model1_type.upper()}"
    model2_name = f"{model2_type.upper()}"
    
    print(f"\n  Model 1: {model1_path} ({model1_name})")
    print(f"  Model 2: {model2_path} ({model2_name})")
    print(f"  Batch size: {args.batch}")
    print(f"  Random seed: {args.seed}")
    
    # Detect precision from first model
    print(f"\n{'─'*40}")
    print("🔍 Detecting model precision...")
    print(f"{'─'*40}")
    
    precision1_str, np_dtype1 = get_model_precision(model1_path)
    precision2_str, np_dtype2 = get_model_precision(model2_path)
    print(f"  Model 1 precision: {precision1_str}")
    print(f"  Model 2 precision: {precision2_str}")
    
    # Use the more precise dtype for input (prefer FP16 if either model uses it)
    if np_dtype1 == np.float16 or np_dtype2 == np.float16:
        np_dtype = np.float16
    else:
        np_dtype = np.float32
    
    # Get input dimensions from first model
    _, channels, height_detected, width_detected = get_model_input_shape(model1_path)
    
    # Handle dimensions
    batch = args.batch
    height = args.height or height_detected
    width = args.width or width_detected
    
    print(f"  Input shape: ({batch}, {channels}, {height}, {width})")
    print(f"  Input dtype: {np_dtype}")
    
    # Create reproducible test input
    print(f"\n{'─'*40}")
    print("🧪 Creating test input...")
    print(f"{'─'*40}")
    
    np.random.seed(args.seed)
    
    # Create normalized input (like real preprocessing)
    # Random values in [0, 1] range, then normalize with ImageNet stats
    raw_input = np.random.rand(batch, channels, height, width).astype(np.float32)
    normalized_input = (raw_input - IMAGENET_MEAN) / IMAGENET_STD
    
    # Convert to model dtype
    input_data = normalized_input.astype(np_dtype)
    
    print(f"  Input range: [{input_data.min():.4f}, {input_data.max():.4f}]")
    print(f"  Input dtype: {input_data.dtype}")
    
    # Run Model 1 inference
    print(f"\n{'─'*40}")
    print(f"🔄 Running {model1_name} inference (Model 1)...")
    print(f"{'─'*40}")
    
    try:
        outputs1 = run_inference(model1_path, input_data)
        print(f"  ✓ Got {len(outputs1)} outputs: {list(outputs1.keys())}")
    except Exception as e:
        print(f"  ❌ {model1_name} inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Run Model 2 inference
    print(f"\n{'─'*40}")
    print(f"🔄 Running {model2_name} inference (Model 2)...")
    print(f"{'─'*40}")
    
    try:
        outputs2 = run_inference(model2_path, input_data)
        print(f"  ✓ Got {len(outputs2)} outputs: {list(outputs2.keys())}")
    except Exception as e:
        print(f"  ❌ {model2_name} inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Compare outputs
    compare_simcc_outputs(outputs1, outputs2, model1_name, model2_name)
    
    print(f"\n{'='*80}")
    print("🏁 DIAGNOSTIC COMPLETE")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
