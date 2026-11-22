"""white_space's input cadence and latency in -sim, over one stretch of a recording.

Runs white_space -sim headless on `--folder`, replayed on the host with the person detector (`camera.replay` and
`yolo` set on the loaded settings file before the app reads them; no settings file is written) and measures while the playhead is between `--start` and `--end`
(seconds of the recording, a stretch found first from the recording's metadata):
- the cameras' hand-over: each camera thread's time per frame (the frame's callbacks, which only queue);
- the syncs: the sets each `CameraSync` hands on, as counts and gaps, against the cameras' frame rate;
- the uploader's time per set;
- the poses per camera out of RAW, and the latencies from a frame's capture to the poses out of RAW and of SMOOTH;
- the tracker's update rate and the light's tick;
- every hand-over's wait (the timers named `Wait …`);
- the drops: the two syncs', the uploader's, the frame source's, the pipeline's, the pose model's.
Writes the report as JSON in `--out` and prints it. The app is stopped in a `finally`, and the run is given up at
twice the time it should take.

    python scripts/white_space_probe.py LABEL --folder FOLDER --start S --end S [--out DIR]
"""
import json
import os
import sys
import threading
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Protocol, cast

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('DEPTHAI_LEVEL', 'error')
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import OpenGL  # noqa: E402
OpenGL.ERROR_CHECKING = False
import numpy as np  # noqa: E402

from modules.logging import setup_logging, install_excepthooks  # noqa: E402
from modules.camera import CameraFrame, FrameType  # noqa: E402
from modules.pose import FrameDict  # noqa: E402
from modules.settings import BaseSettings, SettingsStore  # noqa: E402
from modules.utils import Broadcast, PerformanceTimer  # noqa: E402
from apps.white_space.main import WhiteSpaceMain  # noqa: E402
from apps.white_space.settings import Stage  # noqa: E402


class CallbackCamera(Protocol):
    """A camera as every backend builds it, its frame callbacks a `Broadcast`."""
    frame_callbacks: Broadcast[[CameraFrame]]


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
    parser.add_argument('--record', action='store_true', help='record over the stretch, as `probe_<label>` in the recordings')
    args = parser.parse_args()

    setup_logging(verbose=False, name='white_space')
    install_excepthooks()
    load_startup = SettingsStore.load_startup

    def load_for_replay(self: SettingsStore, root: BaseSettings) -> None:
        """The startup files, then the replay with the detector: both are read when the app builds its cameras."""
        load_startup(self, root)
        root.setup.camera.replay = True
        root.setup.camera.yolo = True
    SettingsStore.load_startup = load_for_replay
    app = WhiteSpaceMain(simulation=True, interface=False)
    SettingsStore.load_startup = load_startup
    settings = app.settings.setup
    settings.camera.simulator.folder = args.folder
    settings.pose.pose.enabled = True

    on = threading.Event()
    camera_thread: dict[int, list[float]] = {}          # ms, per camera
    frames: dict[int, list[float]] = {}                 # each camera's video frames handed over
    syncs: dict[str, list[float]] = {'frames': [], 'tracklets': []}
    uploads: list[float] = []                           # ms per set
    poses: dict[int, int] = {}                          # RAW poses per camera
    latency: dict[str, list[float]] = {'pose_raw': [], 'pose_smooth': []}     # ms
    updates: list[float] = []
    ticks: list[float] = []
    waits: dict[str, list[float]] = {}                  # per hand-over (timer name), ms

    for source in app.cameras:
        camera = cast(CallbackCamera, source)
        callbacks = camera.frame_callbacks

        def hand_over(frame: CameraFrame, callbacks=callbacks) -> None:
            start = time.perf_counter()
            callbacks(frame)
            if on.is_set() and frame.kind == FrameType.VIDEO:
                camera_thread.setdefault(frame.cam_id, []).append((time.perf_counter() - start) * 1000.0)
                frames.setdefault(frame.cam_id, []).append(start)
        camera.frame_callbacks = Broadcast([hand_over], name='probe hand-over')

    def on_set(name: str) -> None:
        if on.is_set():
            syncs[name].append(time.perf_counter())
    app.frame_sync.add_set_callback(lambda _items: on_set('frames'))
    app.tracklet_sync.add_set_callback(lambda _items: on_set('tracklets'))

    upload = app.uploader._upload

    def timed_upload(frames: dict[int, CameraFrame]) -> None:
        start = time.perf_counter()
        upload(frames)
        if on.is_set():
            uploads.append((time.perf_counter() - start) * 1000.0)
    app.uploader._upload = timed_upload

    update = app.tracker.update

    def counted_update(timestamp: float) -> None:
        update(timestamp)
        if on.is_set():
            updates.append(time.perf_counter())
    app.tracker.update = counted_update

    def on_poses(name: str, stage_poses: FrameDict) -> None:
        if on.is_set() and stage_poses:
            latency[name].append((time.perf_counter() - max(frame.timestamp for frame in stage_poses.values())) * 1000.0)
            if name == 'pose_raw':
                for frame in stage_poses.values():
                    poses[frame.cam_id] = poses.get(frame.cam_id, 0) + 1
    app.pipeline.add_frames_callback(Stage.RAW, lambda stage_poses: on_poses('pose_raw', stage_poses))
    app.pipeline.add_frames_callback(Stage.SMOOTH, lambda stage_poses: on_poses('pose_smooth', stage_poses))

    def on_timer(name: str, time_ms: float, _timestamp: float) -> None:
        if on.is_set() and name.startswith('Wait '):
            waits.setdefault(name, []).append(time_ms)
    PerformanceTimer.add_sample_callback(on_timer)

    def on_tick(_tick: object) -> None:
        if on.is_set():
            ticks.append(time.perf_counter())
    app.conductor.add_update_callback(on_tick)

    def playhead() -> float:
        return float(settings.camera.simulator.playhead[0])

    def run() -> None:
        try:
            deadline: float = time.perf_counter() + 2.0 * (STARTUP + args.end)
            # The app starts while the recording plays: once it is up, the playhead is set a second before the stretch
            while float(settings.camera.simulator.playhead[1]) <= 0.0 and time.perf_counter() < deadline:
                time.sleep(0.01)
            settings.camera.simulator.seek = max(0.0, args.start - 1.0) / float(settings.camera.simulator.playhead[1])
            while playhead() >= args.start and time.perf_counter() < deadline:
                time.sleep(0.01)
            while playhead() < args.start and time.perf_counter() < deadline:
                time.sleep(0.01)
            print(f'measuring from playhead {list(settings.camera.simulator.playhead)} in {app.player.active_folder if app.player else None}', flush=True)
            recorder = settings.session.video
            if args.record:
                recorder.name = f'probe_{args.label}'
                recorder.start = True
            on.set()
            began: float = time.perf_counter()
            while args.start <= playhead() < args.end and time.perf_counter() < deadline:
                time.sleep(0.01)
            on.clear()
            ended: float = time.perf_counter()
            print(f'measured until playhead {list(settings.camera.simulator.playhead)}', flush=True)
            if args.record:
                recorder.stop = True
                while recorder.recording and time.perf_counter() < deadline:
                    time.sleep(0.05)
            if ended >= deadline:
                print(f'given up at twice the time it should take ({2.0 * (STARTUP + args.end):.0f} s)', flush=True)
            report = dict(label=args.label, folder=args.folder, stretch=[args.start, args.end], seconds=round(ended - began, 2),
                          camera_thread_ms={cam: spread(v) for cam, v in sorted(camera_thread.items())},
                          frames={cam: gaps(v) for cam, v in sorted(frames.items())},
                          syncs={name: gaps(v) for name, v in syncs.items()}, upload_ms=spread(uploads),
                          poses_per_camera=dict(sorted(poses.items())), latency_ms={k: spread(v) for k, v in latency.items()},
                          tracker_updates=gaps(updates), light_ticks=gaps(ticks),
                          waits_ms={k: spread(v) for k, v in sorted(waits.items())},
                          drops=dict(frame_sync=app.frame_sync.dropped, tracklet_sync=app.tracklet_sync.dropped,
                                     uploader=app.uploader.dropped, pose_step=settings.pose.frame_source.dropped,
                                     pose_pipeline=app.pipeline.dropped, pose_model=settings.pose.pose.dropped))
            if args.record:
                report['recorded'] = dict(missed_tracklets=recorder.missed_tracklets, missed_poses=recorder.missed_poses,
                                          dropped=recorder.dropped)
            out: Path = Path(args.out) / f'white_space_{args.label}.json'
            out.write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2), flush=True)
        finally:
            PerformanceTimer.remove_sample_callback(on_timer)
            app.stop()

    try:
        app.start()
        threading.Thread(target=run, daemon=True).start()
        app.render.run()                  # the window loop on the main thread, until the probe stops the app
        while not app.is_finished:
            time.sleep(0.05)
    finally:
        if not app.is_finished:
            app.stop()
    os._exit(0)


if __name__ == '__main__':
    sys.exit(main())
