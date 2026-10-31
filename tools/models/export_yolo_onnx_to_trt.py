"""Convert a YOLOv8 person detector ONNX model to a TensorRT engine, one per input size.

The size comes from the file's `imgsz` metadata, since ultralytics exports the height and width as
symbols. The engine's precision is the ONNX model's, FP32.

Usage (from the repo root):
    python tools/models/export_yolo_onnx_to_trt.py --onnx data/models/yolov8n_coco_640x512.onnx --output apps/deep_flow/data/models/yolov8n_coco_640x512.trt --opt-batch 1 --max-batch 1
"""
import argparse
import json
from pathlib import Path

import onnx
import tensorrt as trt


def input_size(onnx_path: str) -> tuple[int, int]:
    """The model's input (height, width)."""
    metadata: dict[str, str] = {p.key: p.value for p in onnx.load(onnx_path, load_external_data=False).metadata_props}
    height, width = json.loads(metadata['imgsz'])
    return int(height), int(width)


def export(onnx_path: str, output: str, opt_batch: int, max_batch: int, workspace_gb: int) -> None:
    height, width = input_size(onnx_path)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)     # strongly typed: the precision is the ONNX model's
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError(f"failed to parse {onnx_path}: {[parser.get_error(i) for i in range(parser.num_errors)]}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    profile = builder.create_optimization_profile()
    profile.set_shape(network.get_input(0).name, (1, 3, height, width), (opt_batch, 3, height, width), (max_batch, 3, height, width))
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError(f"failed to build {output}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_bytes(bytes(engine))
    print(f"Exported {onnx_path} to {output} at {width}x{height}, batch 1 to {max_batch}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--opt-batch', type=int, default=1)
    parser.add_argument('--max-batch', type=int, default=1)
    parser.add_argument('--workspace', type=int, default=8, help='workspace size in GB')
    args = parser.parse_args()
    export(args.onnx, args.output, args.opt_batch, args.max_batch, args.workspace)
