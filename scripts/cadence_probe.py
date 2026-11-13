"""deep_flow's input cadence and latency in -sim, over one stretch of a recording.

Runs deep_flow -sim headless on `--folder`, with the recorded tracklets and the pose model running in the app, and
measures while the playhead is between `--start` and `--end` (seconds of the recording, a stretch found first from the
recording's metadata, `scripts/recording_motion.py`):
- the camera's hand-over: the gaps between video frames, and the camera thread's time per frame (the frame's
  callbacks, which only queue);
- the captures the capture sync hands on, and the frames the motion measures, as counts and gaps;
- the render's tick (rate, p95, max);
- the latencies: from a capture's stamp to the capture on the board, and to the poses out of RAW and of SMOOTH;
- every hand-over's wait: how long an item waits in a queue for its receiver (the timers named `Wait …`);
- the uploader's time per color and per depth frame;
- the drops: the uploader's queue, the board's captures, the sync's depth frames without an image.
Writes the report as JSON in `--out` and prints it. The app is stopped in a `finally`, and the run is given up at
twice the time it should take.

    python scripts/cadence_probe.py LABEL --folder FOLDER --start S --end S [--out DIR]
"""
import json
import os
import sys
import threading
import time
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('DEPTHAI_LEVEL', 'error')
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import OpenGL  # noqa: E402
OpenGL.ERROR_CHECKING = False
import numpy as np  # noqa: E402

from modules.log_config import setup_logging, install_excepthooks  # noqa: E402
from modules.camera import CameraFrame, FrameType  # noqa: E402
from modules.pose import FrameDict  # noqa: E402
from modules.render.layers import Motion3DLayer  # noqa: E402
from modules.utils import PerformanceTimer  # noqa: E402
from apps.deep_flow.main import DeepFlowMain  # noqa: E402
from apps.deep_flow.settings import Stage  # noqa: E402

STARTUP: float = 30.0       # s the app takes at most to load its models and start playing


def gaps(times: list[float]) -> dict:
    """A stream of moments: how many, the rate, and the gaps between them (ms)."""
    if len(times) < 2:
        return {'count': len(times)}
    g = np.diff(np.asarray(times)) * 1000.0
    return dict(count=len(times), rate=round(len(g) / (times[-1] - times[0]), 2), mean_ms=round(float(g.mean()), 2),
                p5_ms=round(float(np.percentile(g, 5)), 2), p95_ms=round(float(np.percentile(g, 95)), 2),
                max_ms=round(float(g.max()), 2), over_50ms=int((g > 50.0).sum()))


def spread(values: list[float]) -> dict:
    """Durations (ms): mean, p95, max."""
    if not values:
        return {}
    v = np.asarray(values)
    return dict(count=len(v), mean=round(float(v.mean()), 2), p95=round(float(np.percentile(v, 95)), 2), max=round(float(v.max()), 2))


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('--folder', required=True, help='the recording folder to play')
    parser.add_argument('--start', type=float, required=True, help='the stretch to measure: its start in the recording (s)')
    parser.add_argument('--end', type=float, required=True, help='its end in the recording (s)')
    parser.add_argument('--out', default='.', help='the folder the report goes in')
    args = parser.parse_args()

    setup_logging(verbose=False, name='deep_flow')
    install_excepthooks()
    app = DeepFlowMain(simulation=True, interface=False)
    settings = app.settings
    settings.camera.simulator.folder = args.folder
    settings.camera.person_detector.enabled = False         # the recorded tracklets
    settings.camera.recorded_poses.enabled = False          # the pose model in the app
    settings.pose.pose.enabled = True

    on = threading.Event()
    handover: list[float] = []
    camera_thread: list[float] = []                     # ms
    captures: list[float] = []
    latency: dict[str, list[float]] = {'capture': [], 'pose_raw': [], 'pose_smooth': []}     # ms
    motion: list[float] = []
    ticks: list[float] = []
    uploads: dict[str, list[float]] = {'color': [], 'depth': []}
    waits: dict[str, list[float]] = {}                  # per hand-over (timer name), ms

    camera = app.cameras[0]
    callbacks = list(camera.frame_callbacks)

    def hand_over(frame: CameraFrame) -> None:
        start = time.perf_counter()
        for callback in callbacks:
            callback(frame)
        if on.is_set() and frame.cam_id == 0 and frame.kind == FrameType.VIDEO:
            handover.append(start)
            camera_thread.append((time.perf_counter() - start) * 1000.0)
    camera.frame_callbacks = [hand_over]

    def timed(kind: str, method):
        def wrapper(frame: CameraFrame) -> None:
            start = time.perf_counter()
            method(frame)
            if on.is_set():
                uploads[kind].append((time.perf_counter() - start) * 1000.0)
        return wrapper
    app.uploader._upload = timed('color', app.uploader._upload)
    app.uploader._upload_depth = timed('depth', app.uploader._upload_depth)

    def on_capture(capture) -> None:
        if on.is_set() and capture.cam_id == 0:
            now = time.perf_counter()
            captures.append(now)
            latency['capture'].append((now - capture.timestamp) * 1000.0)
    app.capture_sync.add_capture_callback(on_capture)

    def on_poses(name: str, poses: FrameDict) -> None:
        if on.is_set() and poses:
            latency[name].append((time.perf_counter() - max(frame.timestamp for frame in poses.values())) * 1000.0)
    app.pipeline.add_frames_callback(Stage.RAW, lambda poses: on_poses('pose_raw', poses))
    app.pipeline.add_frames_callback(Stage.SMOOTH, lambda poses: on_poses('pose_smooth', poses))

    def on_timer(name: str, time_ms: float, _timestamp: float) -> None:
        if on.is_set() and name.startswith('Wait '):
            waits.setdefault(name, []).append(time_ms)
    PerformanceTimer.add_sample_callback(on_timer)

    measure = Motion3DLayer._measure

    def counted_measure(layer: Motion3DLayer, input, interval: float) -> None:
        measure(layer, input, interval)
        if on.is_set() and layer._cam_id == 0:
            motion.append(time.perf_counter())
    Motion3DLayer._measure = counted_measure

    def on_tick(now: float) -> None:
        if on.is_set():
            ticks.append(time.perf_counter())
    app.render.add_update_callback(on_tick)

    def playhead() -> float:
        return float(settings.camera.simulator.playhead[0])

    def run() -> None:
        try:
            deadline: float = time.perf_counter() + 2.0 * (STARTUP + args.end)
            while playhead() < args.start and time.perf_counter() < deadline:
                time.sleep(0.01)
            on.set()
            began: float = time.perf_counter()
            while args.start <= playhead() < args.end and time.perf_counter() < deadline:
                time.sleep(0.01)
            on.clear()
            ended: float = time.perf_counter()
            if ended >= deadline:
                print(f'given up at twice the time it should take ({2.0 * (STARTUP + args.end):.0f} s)', flush=True)
            report = dict(label=args.label, folder=args.folder, stretch=[args.start, args.end], seconds=round(ended - began, 2),
                          handover=gaps(handover), captures=gaps(captures), motion=gaps(motion), ticks=gaps(ticks),
                          camera_thread_ms=spread(camera_thread), latency_ms={k: spread(v) for k, v in latency.items()},
                          waits_ms={k: spread(v) for k, v in sorted(waits.items())}, uploads_ms={k: spread(v) for k, v in uploads.items()},
                          drops=dict(uploader=app.uploader.dropped, board=app.board.get_dropped_captures(),
                                     depth_without_image=app.capture_sync.settings.misses))
            out: Path = Path(args.out) / f'cadence_{args.label}.json'
            out.write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2), flush=True)
        finally:
            PerformanceTimer.remove_sample_callback(on_timer)
            app.stop()

    try:
        app.start()
        threading.Thread(target=run, daemon=True).start()
        while not app.is_finished:
            time.sleep(0.05)
    finally:
        if not app.is_finished:
            app.stop()
    os._exit(0)


if __name__ == '__main__':
    sys.exit(main())
