import tensorrt as trt
import numpy as np

print("Starting ONNX to TensorRT conversion...")

# Create builder
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

# Parse ONNX - try FP32 model
print("Parsing ONNX model (FP32)...")
with open("models/rvm_mobilenetv3_fp32.onnx", "rb") as f:
    success = parser.parse(f.read())

if not success:
    print("Failed to parse ONNX file!")
    for i in range(parser.num_errors):
        print(f"Error {i}: {parser.get_error(i)}")
    exit(1)

print("ONNX parsed successfully")

# Configure builder with fixed input shapes
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

# Set fixed input dimensions: src (1, 3, 192, 256)
profile = builder.create_optimization_profile()
profile.set_shape("src", (1, 3, 192, 256), (1, 3, 192, 256), (1, 3, 192, 256))
profile.set_shape("r1i", (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1))
profile.set_shape("r2i", (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1))
profile.set_shape("r3i", (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1))
profile.set_shape("r4i", (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1))
profile.set_shape("downsample_ratio", (1,), (1,), (1,))
config.add_optimization_profile(profile)

config.set_flag(trt.BuilderFlag.FP16)  # TensorRT will still use FP16 internally

# Build engine
print("Building TensorRT engine (this may take 1-2 minutes)...")
serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    print("Failed to build TensorRT engine!")
    exit(1)

print("Engine built successfully")

# Save engine
print("Saving engine to models/rvm_mobilenetv3_fp32.trt...")
with open("models/rvm_mobilenetv3_fp32.trt", "wb") as f:
    f.write(serialized_engine)

print("Conversion complete!")