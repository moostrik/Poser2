"""Inference benchmark: an app's models through their model workers, without cameras or the app.

    python scripts/model_benchmark.py --app white_space
    python scripts/model_benchmark.py --app hd_trio --model-type onnx --reference logs/<earlier>.npz
    python scripts/model_benchmark.py --app deep_flow --models person face

Loads the app's startup preset for the models it enables, their resolutions and its batch size
(the players), feeds the same seeded crops (or `--image`) at the app's crop size one batch at a time,
and times every batch: the worker's own process time and the latency from submit to callback. The
detectors (`person`, `face`, named with `--models`) take one uploaded camera image a run, `--detector-image`
(the face test's photo by default). Writes
`logs/inference_<app>_<type>_<date>_<time>.json` in the timing report's format, so
`scripts/timing_compare.py` compares two of them, with the GPU memory each model takes, and an
`.npz` of the last batch's outputs; `--reference` compares those with an earlier `.npz`.
"""

import importlib
import json
import logging
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Any, Callable

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modules.inference as inference  # noqa: E402
from modules.camera import CameraFrame, FrameType  # noqa: E402
from modules.inference import GpuImage, ImageUpload, ModelType, crop, pose, segmentation  # noqa: E402
from modules.pose import BBox, Frame, FrameDict, Points2D  # noqa: E402
from modules.settings import BaseSettings, presets  # noqa: E402
from modules.utils import PerformanceTimer, Rect, timing  # noqa: E402

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
READY_TIMEOUT = 120.0   # seconds a model may take to load
RESULT_TIMEOUT = 5.0    # seconds a batch may take


def _pose_output(frames: FrameDict) -> dict[str, np.ndarray]:
    ordered = [frames[track_id] for track_id in sorted(frames)]
    return {'points': np.stack([f[Points2D].values for f in ordered]), 'scores': np.stack([f[Points2D].scores for f in ordered])}


def _segmentation_output(images: segmentation.ImageDict) -> dict[str, np.ndarray]:
    ordered = [images[track_id] for track_id in sorted(images)]
    for image in ordered:       # written on the segmentation's stream
        if image.ready is not None:
            torch.cuda.current_stream().wait_event(image.ready)
    return {'mask': torch.stack([i.mask for i in ordered]).float().cpu().numpy(),
            'fgr': torch.stack([i.foreground for i in ordered]).float().cpu().numpy()}


def _person_output(detection: Any) -> dict[str, np.ndarray]:
    return {'boxes': np.array([[b.x, b.y, b.width, b.height] for b in detection.boxes], np.float32).reshape(-1, 4),
            'scores': np.array(detection.scores, np.float32)}


def _face_output(frame: Any) -> dict[str, np.ndarray]:
    return {'boxes': np.array([f.bbox for f in frame.faces], np.float32).reshape(-1, 4),
            'landmarks': np.array([f.landmarks for f in frame.faces], np.float32).reshape(-1, 106, 2)}


def _worker(package: Any, *names: str) -> type:
    """The worker class by its name after E3, or before it, so the script times either commit."""
    for name in names:
        if hasattr(package, name):
            return getattr(package, name)
    raise AttributeError(names)


_person_package: Any = getattr(inference, 'person', None) or getattr(inference, 'detection')

# name (the settings group), the worker class, how its results are registered and read
MODELS: list[tuple[str, type, Callable[[Any, Callable[[Any], None]], None], Callable[[Any], dict[str, np.ndarray]]]] = [
    ('pose', _worker(pose, 'PoseEstimator', 'Predictor'), lambda p, cb: p.add_frames_callback(cb), _pose_output),
    ('segmentation', _worker(segmentation, 'Segmenter', 'Predictor'), lambda p, cb: p.add_segmentation_callback(lambda _poses, images: cb(images)), _segmentation_output),
]
# name (the camera's settings group), the worker class, how its results are registered and read; on a whole image
DETECTORS: list[tuple[str, str, type, Callable[[Any, Callable[[Any], None]], None], Callable[[Any], dict[str, np.ndarray]]]] = [
    ('person', 'person_detector', _person_package.PersonDetector, lambda p, cb: p.add_detection_callback(cb), _person_output),
    ('face', 'face_detector', inference.face.FaceDetector, lambda p, cb: p.add_face_callback(cb), _face_output),
]
DETECTOR_IMAGE: Path = Path('modules/inference/face/tests/data/face.jpg')


def load_app_settings(app: str) -> BaseSettings:
    """The app's setup with its startup preset loaded, as the app's main builds it."""
    presets.set_app(app)
    settings = importlib.import_module(f"apps.{app}.settings").Settings()
    if not presets.load(settings, presets.startup_path()):
        raise FileNotFoundError(f"No startup preset for '{app}' at {presets.startup_path()}")
    setup: BaseSettings = settings.setup
    if app in ('white_space', 'hd_trio'):   # they cut the pose model's crops at its size (their main.py)
        setup.pose.pose_crop.output_width = setup.pose.pose.width
        setup.pose.pose_crop.output_height = setup.pose.pose.height
    settings.initialize()
    return setup


def model_settings(app_group: BaseSettings, model_type: ModelType) -> BaseSettings:
    """A standalone copy of a model's settings group with the requested model type.

    Its model files are the app's: the ONNX files in the repo's `data/models`, the engines in the app's model folder.
    """
    copy = type(app_group)()
    for name in ('max_players', 'resolution', 'onnx_path', 'engine_path'):
        if hasattr(app_group, name):
            setattr(copy, name, getattr(app_group, name))
    copy.model_type = model_type
    copy.initialize()
    return copy


def make_batch(batch: int, width: int, height: int, image: Path | None, seed: int) -> tuple[FrameDict, crop.CropImageDict]:
    """The batch's poses and their crops: (3, H, W) float16 RGB [0, 1] on the GPU, as the crop extractor gives."""
    if image is not None:
        import cv2
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise FileNotFoundError(image)
        rgb = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        base = np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.float32) / 255.0
    else:
        base = np.random.default_rng(seed).random((3, height, width), dtype=np.float32)
    curr = torch.from_numpy(base).to('cuda', torch.float16)
    torch.cuda.synchronize()
    box = BBox.from_rect(Rect(0.0, 0.0, 1.0, 1.0))
    poses: FrameDict = {i: Frame(track_id=i, cam_id=0, timestamp=0.0, features={BBox: box}) for i in range(batch)}
    crops: crop.CropImageDict = {i: crop.Image(crop=curr.clone(), ready=None) for i in range(batch)}
    torch.cuda.synchronize()
    return poses, crops


def gpu_used_mb() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024 / 1024


def run_model(name: str, worker_class: type, add_callback: Callable, read_output: Callable, settings: BaseSettings,
              crop_w: int, crop_h: int, warmup: int, iterations: int, image: Path | None,
              seed: int) -> tuple[dict[str, np.ndarray], float, dict[str, dict[str, float]]]:
    """Time one model; returns the last batch's outputs, the GPU memory its worker took (MB) and the timers' stats
    (the worker's own and the latency).

    The stats are read here, while the timers live: the session holds them weakly.
    """
    label = f"{name} {settings.model_type.name} {settings.model} b{settings.max_players}"
    latency_timer = PerformanceTimer(name=f"{label} latency")

    memory_before = gpu_used_mb()
    worker = worker_class(settings)
    result: dict[str, Any] = {}
    done = Event()

    def on_output(output: Any) -> None:
        result['output'] = output
        done.set()

    add_callback(worker, on_output)
    worker.start()
    outputs: dict[str, np.ndarray] = {}
    try:
        deadline = time.perf_counter() + READY_TIMEOUT
        while not worker.is_ready:
            if time.perf_counter() > deadline:
                raise RuntimeError(f"{label}: model not ready ({settings.model_file(settings.model)})")
            time.sleep(0.05)
        memory_mb = gpu_used_mb() - memory_before

        poses, crops = make_batch(settings.max_players, crop_w, crop_h, image, seed)
        for i in range(warmup + iterations):
            done.clear()
            start = time.perf_counter()
            worker.submit(poses, crops)
            if not done.wait(RESULT_TIMEOUT):
                raise RuntimeError(f"{label}: no result for batch {i + 1}")
            latency_ms = (time.perf_counter() - start) * 1000.0
            if i == warmup:
                PerformanceTimer.start_session()
            if i >= warmup:
                latency_timer.add_time(latency_ms, report=False)
        outputs = {f"{name}/{k}": v for k, v in read_output(result['output']).items()}
    finally:
        worker.stop()
    return outputs, memory_mb, PerformanceTimer.stop_session()


def detector_settings(app_group: BaseSettings, model_type: ModelType) -> BaseSettings:
    """A standalone copy of a detector's settings with the requested model type: its model files as the app's, its
    thresholds as the preset's, on."""
    copy = type(app_group)()
    for name in ('resolution', 'onnx_path', 'engine_path', 'confidence', 'overlap', 'threshold', 'pitch_zero'):
        if hasattr(app_group, name):
            setattr(copy, name, getattr(app_group, name))
    copy.model_type = model_type
    if hasattr(copy, 'enabled'):
        copy.enabled = True
    copy.initialize()
    return copy


def uploaded_image(path: Path) -> GpuImage:
    """A camera image as the uploader hands it on, on the GPU with its written event."""
    import cv2
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    frame = CameraFrame(0, FrameType.VIDEO, bgr, 1.0)
    images, ready = ImageUpload().upload([frame], torch.cuda.Stream())
    torch.cuda.synchronize()
    return GpuImage(0, 1.0, images[0], ready, frame)


def run_detector(name: str, worker_class: type, add_callback: Callable, read_output: Callable, settings: BaseSettings,
                 warmup: int, iterations: int, image_path: Path) -> tuple[dict[str, np.ndarray], float, dict[str, dict[str, float]]]:
    """Time one detector on one uploaded image a run, as `run_model` times a crop model."""
    label = f"{name} {settings.model_type.name} image"
    latency_timer = PerformanceTimer(name=f"{label} latency")
    memory_before = gpu_used_mb()
    worker = worker_class(settings)
    result: dict[str, Any] = {}
    done = Event()

    def on_output(output: Any) -> None:
        result['output'] = output
        done.set()

    add_callback(worker, on_output)
    worker.start()
    outputs: dict[str, np.ndarray] = {}
    try:
        deadline = time.perf_counter() + READY_TIMEOUT
        while not worker.is_ready:
            if time.perf_counter() > deadline:
                raise RuntimeError(f"{label}: model not ready")
            time.sleep(0.05)
        memory_mb = gpu_used_mb() - memory_before
        image: GpuImage = uploaded_image(image_path)
        for i in range(warmup + iterations):
            done.clear()
            start = time.perf_counter()
            worker.submit(image)
            if not done.wait(RESULT_TIMEOUT):
                raise RuntimeError(f"{label}: no result for image {i + 1}")
            latency_ms = (time.perf_counter() - start) * 1000.0
            if i == warmup:
                PerformanceTimer.start_session()
            if i >= warmup:
                latency_timer.add_time(latency_ms, report=False)
        outputs = {f"{name}/{k}": v for k, v in read_output(result['output']).items()}
    finally:
        worker.stop()
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
    assert __doc__ is not None
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--app', required=True, choices=['white_space', 'hd_trio', 'deep_flow', 'low_flow', 'smile'])
    parser.add_argument('--model-type', default='trt', choices=['trt', 'onnx'])
    parser.add_argument('--iterations', type=int, default=1000, help='timed batches per model')
    parser.add_argument('--warmup', type=int, default=100, help='untimed batches per model first')
    parser.add_argument('--image', type=Path, default=None, help='a photo to crop from instead of seeded noise')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--models', nargs='*', default=None, help='the models to run, enabled or not (pose, segmentation, person, face); default the enabled crop models')
    parser.add_argument('--detector-image', type=Path, default=DETECTOR_IMAGE, help='the photo the detectors run on')
    parser.add_argument('--reference', type=Path, default=None, help='an earlier .npz to compare the outputs with')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s %(message)s")

    model_type = ModelType.ONNX if args.model_type == 'onnx' else ModelType.TRT
    app_settings = load_app_settings(args.app)
    stats: dict[str, dict[str, float]] = {}
    memory: dict[str, float] = {}
    outputs: dict[str, np.ndarray] = {}
    for name, worker_class, add_callback, read_output in MODELS:
        group = getattr(getattr(app_settings, 'pose', None), name, None)
        if group is None or (args.models is None and not group.enabled) or (args.models is not None and name not in args.models):
            continue
        # the segmentation's own crops where the app cuts them, else the pose model's
        crop_settings = getattr(app_settings.pose, f"{name}_crop", None) or app_settings.pose.pose_crop
        settings = model_settings(group, model_type)
        logger.info("%s: %s, batch %d, crop %dx%d", name, settings.model, settings.max_players,
                    crop_settings.output_width, crop_settings.output_height)
        model_outputs, memory_mb, model_stats = run_model(name, worker_class, add_callback, read_output, settings,
                                                          crop_settings.output_width, crop_settings.output_height,
                                                          args.warmup, args.iterations, args.image, args.seed)
        stats.update(model_stats)
        memory[name] = memory_mb
        outputs.update(model_outputs)

    for name, group_name, worker_class, add_callback, read_output in DETECTORS:
        group = getattr(getattr(app_settings, 'camera', None), group_name, None)
        if group is None or args.models is None or name not in args.models:
            continue
        settings = detector_settings(group, model_type)
        logger.info("%s: %s, image %s", name, type(settings).__name__, args.detector_image)
        model_outputs, memory_mb, model_stats = run_detector(name, worker_class, add_callback, read_output, settings,
                                                             args.warmup, args.iterations, args.detector_image)
        stats.update(model_stats)
        memory[name] = memory_mb
        outputs.update(model_outputs)

    LOG_DIR.mkdir(exist_ok=True)
    stem = f"inference_{args.app}_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    np.savez_compressed(LOG_DIR / f"{stem}.npz", **outputs)  # pyright: ignore[reportArgumentType]
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
