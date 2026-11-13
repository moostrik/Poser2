"""Convert the face models to TensorRT engines: SCRFD (`scrfd_10g_bnkps.onnx`) at its 640 x 640 input, batch 1, and
the 106-point landmarks (`2d106det.onnx`) at 192 x 192 for a batch of faces.

The landmarks model runs a batch but declares its output at batch 1; its batch is made dynamic before the build.

Usage (from the repo root):
    python tools/models/export_face_onnx_to_trt.py --onnx data/models/scrfd_10g_bnkps.onnx --output apps/smile/data/models/scrfd_10g_bnkps.trt --size 640
    python tools/models/export_face_onnx_to_trt.py --onnx data/models/2d106det.onnx --output apps/smile/data/models/2d106det.trt --size 192 --opt-batch 2 --max-batch 8
"""
import argparse
from pathlib import Path

import onnx
import tensorrt as trt


def with_dynamic_batch(model: onnx.ModelProto) -> onnx.ModelProto:
    """The model with the first dimension of its inputs and outputs named, so TensorRT builds them for a batch."""
    for value in list(model.graph.input) + list(model.graph.output):
        dims = value.type.tensor_type.shape.dim
        if len(dims) > 0:
            dims[0].ClearField('dim_value')
            dims[0].dim_param = 'batch'
    return model


def export(onnx_path: str, output: str, size: int, opt_batch: int, max_batch: int, workspace_gb: int) -> None:
    loaded: onnx.ModelProto = onnx.load(onnx_path)
    # SCRFD's outputs are per anchor (12800 x 1, ...), not per image: only a batched model's first dimension is the batch
    model: bytes = (with_dynamic_batch(loaded) if max_batch > 1 else loaded).SerializeToString()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(model):
        raise RuntimeError(f"failed to parse {onnx_path}: {[parser.get_error(i) for i in range(parser.num_errors)]}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    profile = builder.create_optimization_profile()
    profile.set_shape(network.get_input(0).name, (1, 3, size, size), (opt_batch, 3, size, size), (max_batch, 3, size, size))
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError(f"failed to build {output}")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_bytes(bytes(engine))
    print(f"Exported {onnx_path} to {output} at {size}x{size}, batch 1 to {max_batch}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--size', type=int, required=True, help='the square input size')
    parser.add_argument('--opt-batch', type=int, default=1)
    parser.add_argument('--max-batch', type=int, default=1)
    parser.add_argument('--workspace', type=int, default=8, help='workspace size in GB')
    args = parser.parse_args()
    export(args.onnx, args.output, args.size, args.opt_batch, args.max_batch, args.workspace)
