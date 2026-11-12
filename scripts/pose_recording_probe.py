"""deep_flow's recorded poses in -sim, two runs.

record: plays the preset's recording with the pose model on (`recorded_poses.enabled` off), records it again for
`--seconds`, and reports what the new recording holds: its frames, those with poses, the poses' latency, the
recorder's counts.

play FOLDER: plays a recording with its recorded poses (`recorded_poses.enabled` on) for `--seconds`, and reports
the RAW stage's poses against the recording's: how many came, how many match the recorded box and points, how late
after their capture, and the track_ids the recording had not.

    python scripts/pose_recording_probe.py record [--seconds 12]
    python scripts/pose_recording_probe.py play FOLDER [--seconds 12]
"""
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
from modules.camera import read_metadata  # noqa: E402
from modules.pose import BBox, FrameDict, Points2D  # noqa: E402
from apps.deep_flow.main import DATA_PATH, DeepFlowMain  # noqa: E402
from apps.deep_flow.settings import Stage  # noqa: E402

RECORDINGS: Path = ROOT / DATA_PATH / 'recordings'
NAME: str = 'poses_probe'


def percentiles(values: list[float]) -> str:
    if not values:
        return 'none'
    v = np.asarray(values) * 1000.0
    return f'median {np.median(v):.1f} ms, p95 {np.percentile(v, 95):.1f} ms, max {v.max():.1f} ms ({len(v)})'


def record(app: DeepFlowMain, seconds: float) -> None:
    app.settings.camera.recorded_poses.enabled = False      # the pose model runs
    app.settings.camera.recorder.name = NAME
    time.sleep(3.0)                                         # the player loads and the model warms up
    before: set[str] = {p.name for p in RECORDINGS.iterdir()}
    app.settings.camera.start = True
    time.sleep(seconds)
    app.settings.camera.stop = True
    time.sleep(3.0)                                         # the chunk merges
    recorder = app.settings.camera.recorder
    print(f'recorder counts: depth {recorder.missed_depth}, color {recorder.missed_color}, '
          f'tracklets {recorder.missed_tracklets}, poses {recorder.missed_poses}')
    new = sorted(p for p in RECORDINGS.iterdir() if p.name not in before)
    for folder in new:
        for chunk in sorted(folder.glob('*.mkv')):
            frames = read_metadata(chunk).frames
            posed = [f for f in frames if f.poses is not None]
            tracked = [f for f in frames if f.tracklets]
            print(f'{folder.name}/{chunk.name}: {len(frames)} frames, {len(tracked)} with players, {len(posed)} with poses')
            print('  pose latency', percentiles([f.pose_latency for f in posed if f.pose_latency is not None]))
            print('  tracklet latency', percentiles([f.tracklet_latency for f in frames if f.tracklet_latency is not None]))


def play(app: DeepFlowMain, seconds: float) -> None:
    app.settings.camera.recorded_poses.enabled = True
    node = app.recorded_poses
    assert node is not None
    came: list[tuple[float, FrameDict, dict]] = []      # arrival, the RAW poses, the recording's of their frame then

    def on_raw(frames: FrameDict) -> None:
        recorded: dict = {}
        with node._lock:
            for frame in frames.values():
                recorded[frame.track_id] = node._recorded[frame.cam_id].at(frame.timestamp)
        came.append((time.perf_counter(), dict(frames), recorded))
    app.pipeline.add_frames_callback(Stage.RAW, on_raw)
    time.sleep(seconds)
    matched, differing, without, lags = 0, 0, 0, []
    for arrival, frames, recorded in came:
        for frame in frames.values():
            metadata = recorded[frame.track_id]
            if metadata is None or metadata.poses is None or frame.track_id not in metadata.poses:
                without += 1
                continue
            box, points = metadata.poses[frame.track_id]
            rect = frame[BBox].to_rect()
            same = (np.allclose([rect.x, rect.y, rect.width, rect.height], box, atol=1e-4)
                    and np.allclose(frame[Points2D].values, points[:, :2], atol=1e-4, equal_nan=True))
            matched += same
            differing += not same
            lags.append(arrival - frame.timestamp)
    print(f'RAW poses: {sum(len(f) for _, f, _ in came)} in {len(came)} frames; {matched} match the recording, '
          f'{differing} differ, {without} without a recorded pose; track_ids missed {node.missed.total("track_id")}')
    print('  after their capture', percentiles(lags))


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument('phase', choices=['record', 'play'])
    parser.add_argument('folder', nargs='?', default='')
    parser.add_argument('--seconds', type=float, default=12.0)
    parser.add_argument('--bgr', action='store_true', help='decode the color to BGR, as before the player handed NV12')
    args = parser.parse_args()

    setup_logging(verbose=False, name='deep_flow')
    install_excepthooks()
    app = DeepFlowMain(simulation=True, interface=False)
    if args.bgr:
        assert app.player is not None
        app.player.nv12 = False
    if args.folder:
        app.settings.camera.simulator.folder = args.folder

    def run() -> None:
        try:
            if args.phase == 'record':
                record(app, args.seconds)
            else:
                play(app, args.seconds)
        finally:
            app.stop()

    app.start()
    threading.Thread(target=run, daemon=True).start()
    deadline: float = time.perf_counter() + 2 * (args.seconds + 10.0)
    while not app.is_finished and time.perf_counter() < deadline:
        time.sleep(0.05)
    if not app.is_finished:
        print('stopped at twice the time it should take', flush=True)
    os._exit(0)


if __name__ == '__main__':
    sys.exit(main())
