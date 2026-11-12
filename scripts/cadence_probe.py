"""deep_flow's input cadence in -sim: does every camera frame get through to the motion, and how fast?

Runs deep_flow -sim headless (the preset's recording) and, per whole loop of the recording, measures:
- the camera's hand-over: the gaps between video frames, and the camera thread's time per frame (the frame's
  callbacks, which only queue);
- the captures the capture sync hands on, and the frames the motion measures, as counts and gaps;
- the render's tick (rate, p95, max);
- the capture latency, from the frame's stamp to the capture on the board;
- the uploader's time per color and per depth frame;
- the drops: the uploader's queue, the board's captures, the sync's depth frames without an image.
Writes the report as JSON next to `--out` and prints one line per loop.

    python scripts/cadence_probe.py LABEL [--loops 2] [--warmup 15] [--out DIR]
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
from modules.render.layers import Motion3DLayer  # noqa: E402
from apps.deep_flow.main import DeepFlowMain  # noqa: E402


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
    parser.add_argument('--loops', type=int, default=2, help='whole loops of the recording to measure, each from its start to the next')
    parser.add_argument('--warmup', type=float, default=15.0, help='seconds before measuring')
    parser.add_argument('--out', default='.', help='the folder the report goes in')
    args = parser.parse_args()

    setup_logging(verbose=False, name='deep_flow')
    install_excepthooks()
    app = DeepFlowMain(simulation=True, interface=False)
    on = threading.Event()
    handover: list[float] = []
    camera_thread: list[tuple[float, float]] = []      # moment, ms
    captures: list[float] = []
    latency: list[tuple[float, float]] = []            # moment, ms
    motion: list[float] = []
    ticks: list[float] = []
    uploads: dict[str, list[float]] = {'color': [], 'depth': []}
    loop_starts: list[float] = []

    camera = app.cameras[0]
    callbacks = list(camera.frame_callbacks)

    def hand_over(frame: CameraFrame) -> None:
        start = time.perf_counter()
        for callback in callbacks:
            callback(frame)
        if on.is_set() and frame.cam_id == 0 and frame.kind == FrameType.VIDEO:
            handover.append(start)
            camera_thread.append((start, (time.perf_counter() - start) * 1000.0))
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
            latency.append((now, (now - capture.timestamp) * 1000.0))
    app.capture_sync.add_capture_callback(on_capture)

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

    def on_loop() -> None:
        now = time.perf_counter()
        if not loop_starts or now - loop_starts[-1] > 1.0:      # a loop's start can call back twice
            loop_starts.append(now)
    assert app.player is not None
    app.player.add_loop_callback(on_loop)

    def section(start: float, end: float) -> dict:
        within = lambda times: [t for t in times if start <= t < end]                  # noqa: E731
        return dict(seconds=round(end - start, 2), handover=gaps(within(handover)), captures=gaps(within(captures)),
                    motion=gaps(within(motion)), ticks=gaps(within(ticks)),
                    camera_thread_ms=spread([v for t, v in camera_thread if start <= t < end]),
                    capture_latency_ms=spread([v for t, v in latency if start <= t < end]))

    def run() -> None:
        time.sleep(args.warmup)
        on.set()
        seen: int = len(loop_starts)
        deadline: float = time.perf_counter() + 30.0 * (args.loops + 1)
        while len(loop_starts) < seen + args.loops + 1 and time.perf_counter() < deadline:
            time.sleep(0.05)
        on.clear()
        starts: list[float] = loop_starts[seen:]
        sections = {f'loop_{i + 1}': section(starts[i] + 0.5, starts[i + 1]) for i in range(len(starts) - 1)}   # past the loop's restart
        report = dict(label=args.label, sections=sections, uploads={kind: spread(v) for kind, v in uploads.items()},
                      drops=dict(uploader=app.uploader.dropped, board=app.board.get_dropped_captures(),
                                 depth_without_image=app.capture_sync.settings.misses))
        out: Path = Path(args.out) / f'cadence_{args.label}.json'
        out.write_text(json.dumps(report, indent=2))
        for name, s in sections.items():
            h, c, m, t = s['handover'], s['captures'], s['motion'], s['ticks']
            print(f"{name}: motion {m.get('count')}/{h.get('count')} | captures {c.get('count')} | ticks {t.get('rate')}/s "
                  f"p95 {t.get('p95_ms')} max {t.get('max_ms')} | camera thread {s['camera_thread_ms'].get('mean')}/"
                  f"{s['camera_thread_ms'].get('p95')} ms | latency {s['capture_latency_ms'].get('mean')}/{s['capture_latency_ms'].get('p95')} ms")
        print('uploads', report['uploads'], 'drops', report['drops'], f'-> {out}', flush=True)
        app.stop()

    app.start()
    threading.Thread(target=run, daemon=True).start()
    while not app.is_finished:
        time.sleep(0.05)
    os._exit(0)


if __name__ == '__main__':
    sys.exit(main())
