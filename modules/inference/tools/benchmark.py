"""Inference benchmark: an app's models through the app's runners, without cameras or the app.

    python -m modules.inference.tools.benchmark --app white_space
    python -m modules.inference.tools.benchmark --app hd_trio --model-type onnx --reference logs/<earlier>.npz

Loads the app's startup preset for the models it enables, their resolutions and its batch size
(the players), feeds the same seeded crops (or `--image`) at the app's crop size, and times every
batch: the runner's own inference time and the latency from submit to callback. Writes
`logs/inference_<app>_<type>_<date>_<time>.json` in the timing report's format, so
`scripts/timing_compare.py` compares two of them, with the GPU memory each model takes, and an
`.npz` of the last batch's outputs; `--reference` compares those with an earlier `.npz`.
"""

import importlib
import json
import logging
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Any, Callable

import numpy as np
import torch

from modules.settings import presets
from modules.utils import PerformanceTimer, timing

from .. import ModelType
from ..optical_flow.io import OpticalFlowInput
from ..pose.io import DetectionInput
from ..segmentation.io import SegmentationInput

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).resolve().parents[3] / "logs"
READY_TIMEOUT = 120.0   # seconds a runner may take to load its model
RESULT_TIMEOUT = 5.0    # seconds a batch may take


@dataclass
class Model:
    """One model: its name (the settings group and the subpackage of its runners), how it is fed and read."""
    name: str
    make_input: Callable[[int, list[torch.Tensor], list[torch.Tensor]], Any]
    read_output: Callable[[Any], dict[str, np.ndarray]]


def _pose_output(output: Any) -> dict[str, np.ndarray]:
    return {'points': np.stack(output.point_batch), 'scores': np.stack(output.score_batch)}


def _segmentation_output(output: Any) -> dict[str, np.ndarray]:
    return {'mask': output.mask_tensor.float().cpu().numpy(), 'fgr': output.fgr_tensor.float().cpu().numpy()}


def _flow_output(output: Any) -> dict[str, np.ndarray]:
    return {'flow': output.flow_tensor.float().cpu().numpy()}


MODELS: list[Model] = [
    Model('pose',
          lambda batch_id, crops, _prev: DetectionInput(batch_id=batch_id, gpu_images=crops), _pose_output),
    Model('segmentation',
          lambda batch_id, crops, _prev: SegmentationInput(batch_id=batch_id, gpu_images=crops,
                                                           track_ids=list(range(len(crops)))), _segmentation_output),
    Model('optical_flow',
          lambda batch_id, crops, prev: OpticalFlowInput(batch_id=batch_id, gpu_image_pairs=list(zip(prev, crops)),
                                                         track_ids=list(range(len(crops)))), _flow_output),
]


def load_app_settings(app: str) -> Any:
    """The app's settings with its startup preset loaded, as the app's main builds them."""
    presets.set_app(app)
    settings = importlib.import_module(f"apps.{app}.settings").Settings()
    if not presets.load(settings, presets.startup_path()):
        raise FileNotFoundError(f"No startup preset for '{app}' at {presets.startup_path()}")
    if app == 'white_space':   # White Space cuts its crops at the pose model's size (apps/white_space/main.py)
        settings.pose.image_crop.output_width = settings.pose.pose.width
        settings.pose.image_crop.output_height = settings.pose.pose.height
    settings.initialize()
    return settings


def model_settings(app_group: Any, model_type: ModelType) -> Any:
    """A standalone copy of a model's settings group with the requested model type.

    The ONNX models are named from the repo root (`data/models/...`), the TensorRT engines from the app's
    model folder.
    """
    copy = type(app_group)()
    copy.max_players = app_group.max_players
    copy.resolution = app_group.resolution
    copy.model_type = model_type
    copy.model_path = '.' if model_type is ModelType.ONNX else app_group.model_path
    copy.initialize()
    return copy


def make_crops(batch: int, width: int, height: int, image: Path | None, seed: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """The batch's crops and the previous frame's: (3, H, W) float16 RGB [0, 1] on the GPU, as the crop extractor gives."""
    if image is not None:
        import cv2
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise FileNotFoundError(image)
        rgb = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        base = np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.float32) / 255.0
        frames = [base, np.roll(base, 4, axis=2)]
    else:
        rng = np.random.default_rng(seed)
        frames = [rng.random((3, height, width), dtype=np.float32) for _ in range(2)]
    curr = torch.from_numpy(frames[0]).to('cuda', torch.float16)
    prev = torch.from_numpy(frames[1]).to('cuda', torch.float16)
    return [curr.clone() for _ in range(batch)], [prev.clone() for _ in range(batch)]


def gpu_used_mb() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024 / 1024


def run_model(model: Model, settings: Any, crop_w: int, crop_h: int, warmup: int, iterations: int,
              image: Path | None, seed: int) -> tuple[dict[str, np.ndarray], float, dict[str, dict[str, float]]]:
    """Time one model; returns the last batch's outputs, the GPU memory its runner took (MB) and its timer stats.

    The stats are read here, while the timers live: the session holds them weakly.
    """
    runner_module = importlib.import_module(f"..{model.name}.runner_"
                                            f"{'onnx' if settings.model_type is ModelType.ONNX else 'trt'}", __package__)
    runner_class = getattr(runner_module, 'RunnerONNX' if settings.model_type is ModelType.ONNX else 'RunnerTRT')

    label = f"{model.name} {settings.model_type.name} {settings.resolution.name} b{settings.max_players}"
    inference_timer = PerformanceTimer(name=f"{label} inference")
    latency_timer = PerformanceTimer(name=f"{label} latency")

    memory_before = gpu_used_mb()
    runner = runner_class(settings)
    result: dict[str, Any] = {}
    done = Event()

    def on_output(output: Any) -> None:
        result['output'] = output
        done.set()

    runner.register_callback(on_output)
    runner.start()
    deadline = time.monotonic() + READY_TIMEOUT
    while not runner.is_ready:
        if time.monotonic() > deadline or not runner.is_alive():
            runner.stop()
            raise RuntimeError(f"{label}: runner not ready (model {settings.model_path}/{settings.model})")
        time.sleep(0.05)
    memory_mb = gpu_used_mb() - memory_before

    crops, prev = make_crops(settings.max_players, crop_w, crop_h, image, seed)
    outputs: dict[str, np.ndarray] = {}
    try:
        for i in range(warmup + iterations):
            done.clear()
            start = time.perf_counter()
            runner.submit(model.make_input(i + 1, crops, prev))
            if not done.wait(RESULT_TIMEOUT):
                raise RuntimeError(f"{label}: no result for batch {i + 1}")
            latency_ms = (time.perf_counter() - start) * 1000.0
            output = result['output']
            if not output.processed:
                raise RuntimeError(f"{label}: batch {i + 1} was not processed")
            if i == warmup:
                PerformanceTimer.start_session()
            if i >= warmup:
                inference_timer.add_time(output.inference_time_ms, report=False)
                latency_timer.add_time(latency_ms, report=False)
        outputs = {f"{model.name}/{k}": v for k, v in model.read_output(result['output']).items()}
    finally:
        runner.stop()
    return outputs, memory_mb, PerformanceTimer.stop_session()


def compare_outputs(outputs: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Max and mean absolute difference per output that both have."""
    agreement: dict[str, dict[str, float]] = {}
    for name in sorted(set(outputs) & set(reference)):
        if outputs[name].shape != reference[name].shape:
            agreement[name] = {'shape_mismatch': 1.0}
            continue
        diff = np.abs(outputs[name].astype(np.float64) - reference[name].astype(np.float64))
        agreement[name] = {'max_abs': float(diff.max()), 'mean_abs': float(diff.mean())}
    return agreement


def main() -> None:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--app', required=True, choices=['white_space', 'hd_trio', 'deep_flow'])
    parser.add_argument('--model-type', default='trt', choices=['trt', 'onnx'])
    parser.add_argument('--iterations', type=int, default=1000, help='timed batches per model')
    parser.add_argument('--warmup', type=int, default=100, help='untimed batches per model first')
    parser.add_argument('--image', type=Path, default=None, help='a photo to crop from instead of seeded noise')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--reference', type=Path, default=None, help='an earlier .npz to compare the outputs with')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s %(message)s")

    model_type = ModelType.ONNX if args.model_type == 'onnx' else ModelType.TRT
    app_settings = load_app_settings(args.app)
    crop = app_settings.pose.image_crop
    stats: dict[str, dict[str, float]] = {}
    memory: dict[str, float] = {}
    outputs: dict[str, np.ndarray] = {}
    for model in MODELS:
        group = getattr(app_settings.pose, model.name, None)
        if group is None or not group.enabled:
            continue
        settings = model_settings(group, model_type)
        logger.info("%s: %s, batch %d, crop %dx%d", model.name, settings.model, settings.max_players,
                    crop.output_width, crop.output_height)
        model_outputs, memory_mb, model_stats = run_model(model, settings, crop.output_width, crop.output_height,
                                             args.warmup, args.iterations, args.image, args.seed)
        stats.update(model_stats)
        memory[model.name] = memory_mb
        outputs.update(model_outputs)

    LOG_DIR.mkdir(exist_ok=True)
    stem = f"inference_{args.app}_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    np.savez_compressed(LOG_DIR / f"{stem}.npz", **outputs)
    report = {
        'app': args.app,
        'model_type': args.model_type,
        'iterations': args.iterations,
        'warmup': args.warmup,
        'input': str(args.image) if args.image else f"seed {args.seed}",
        'commit': timing.git_commit(),
        'gpu': torch.cuda.get_device_name(),
        'packages': timing.package_versions(),
        'gpu_memory_mb': memory,
        'timers': stats,
    }
    if args.reference is not None:
        with np.load(args.reference) as reference:
            report['agreement'] = compare_outputs(outputs, dict(reference))
        for name, values in report['agreement'].items():
            logger.info("agreement %-24s %s", name, ", ".join(f"{k} {v:.5f}" for k, v in values.items()))
    (LOG_DIR / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, values in stats.items():
        logger.info("%-44s mean %7.3f  p50 %7.3f  p99 %7.3f ms", name, values['mean'], values['p50'], values['p99'])
    logger.info("Report: %s", LOG_DIR / f"{stem}.json")


if __name__ == '__main__':
    main()
