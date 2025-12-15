# type: ignore

import tensorrt as trt
import numpy as np
import os

print("Starting RTMPose ONNX to TensorRT conversion...")

# Parse ONNX model
onnx_path = "models/rtmpose-l_256x192.onnx"

# Check if file exists
if not os.path.exists(onnx_path):
    print(f"ERROR: ONNX file not found at: {onnx_path}")
    print("Please check the file path and name.")
    exit(1)

print(f"Found ONNX file: {onnx_path}")

# Create builder
print("Creating TensorRT builder...")
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

print("Parsing RTMPose ONNX model...")
with open(onnx_path, "rb") as f:
    model_data = f.read()
    print(f"Read {len(model_data)} bytes from ONNX file")
    success = parser.parse(model_data)

if not success:
    print("Failed to parse ONNX file!")
    for i in range(parser.num_errors):
        print(f"Error {i}: {parser.get_error(i)}")
    exit(1)

print("ONNX parsed successfully")
print(f"Network has {network.num_inputs} inputs and {network.num_outputs} outputs")

# Configure builder with fixed input shapes for 256x192 input
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB workspace

# RTMPose input: (batch, 3, height, width) = (1, 3, 256, 192)
profile = builder.create_optimization_profile()
profile.set_shape("input", (1, 3, 256, 192), (1, 3, 256, 192), (1, 3, 256, 192))
config.add_optimization_profile(profile)

# Enable FP16 for faster inference
config.set_flag(trt.BuilderFlag.FP16)

# Build engine
print("Building TensorRT engine (this may take 2-5 minutes)...")
print("Please wait, this process can take a while...")
serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    print("Failed to build TensorRT engine!")
    exit(1)

print("Engine built successfully")

# Save engine
output_path = "models/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230126.trt"
print(f"Saving engine to {output_path}...")
with open(output_path, "wb") as f:
    f.write(serialized_engine)

print("Conversion complete!")
print(f"TensorRT engine saved to: {output_path}")