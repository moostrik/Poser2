"""What the recording tests share: frames, tracklets and poses to record, a recorder and a player
set up for the tests, and a recording made in a folder."""

import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np

from modules import video
from modules.camera import CameraFrame, CameraTracklet, CameraTracklets, FrameMetadata, FramePoses, FrameReadings, \
    FrameType, Player, PlayerSettings, Recorder, RecorderSettings, TrackingStatus
from modules.camera.recording.recorder.recorder import RecState
from modules.utils import Rect


LENS: list[list[float]] = [[538.0, 0.0, 450.0], [0.0, 538.0, 360.0], [0.0, 0.0, 1.0]]


def video_frame(i: int, timestamp: float | None = None) -> CameraFrame:
    """Frame `i`, captured at `i` s unless given."""
    t: float = float(i) if timestamp is None else timestamp
    return CameraFrame(0, FrameType.VIDEO, np.full((720, 896, 3), i % 255, np.uint8), t, readings=FrameReadings(-30.0 + i, 1.0, 1000, 100),
                       intrinsics=LENS, arrival=t + 0.01)


POINTS: np.ndarray = np.tile(np.array([[0.25, 0.5, 0.9]], dtype=np.float32), (17, 1))


def poses(timestamp: float) -> FramePoses:
    """The host's poses of camera 0's frame captured at `timestamp`: one player, track 3."""
    return FramePoses(0, timestamp, {3: (np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), POINTS)}, timestamp + 0.05)


def tracklets(timestamp: float) -> CameraTracklets:
    tracklet = CameraTracklet(cam_id=0, id=1, status=TrackingStatus.TRACKED, roi=Rect(0.1, 0.2, 0.3, 0.4), age=1, timestamp=timestamp)
    return CameraTracklets(0, timestamp, [tracklet], arrival=timestamp + 0.033)


def depth_frame(value: int, timestamp: float) -> CameraFrame:
    return CameraFrame(0, FrameType.DEPTH, np.full((300, 374), value, np.uint16), timestamp)


def recorder(types: list[FrameType], temp: str = 'temp', output: str = 'out', data_path: str = '', keyframe_interval: int = 1) -> Recorder:
    settings = RecorderSettings()
    settings.video_frame_types = types
    settings.video_encoder = video.CoderType.iGPU
    settings.temp_path = temp
    settings.output_path = output
    settings.fps = 60.0
    settings.keyframe_interval = keyframe_interval
    return Recorder(settings, data_path)


def stream_frames(path: Path, stream: int) -> int:
    """The frames actually stored in one stream of a chunk."""
    out = subprocess.run(['ffprobe', '-hide_banner', '-loglevel', 'error', '-count_packets', '-select_streams', f'v:{stream}',
                          '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', str(path)], capture_output=True, text=True).stdout
    return int(out.strip().strip(','))


def cam_frame(cam_id: int, value: int, timestamp: float) -> CameraFrame:
    """A small color frame of a camera, its pixels `value`, captured at `timestamp`."""
    return CameraFrame(cam_id, FrameType.VIDEO, np.full((240, 320, 3), value, np.uint8), timestamp, readings=FrameReadings(0.0, 1.0, 1000, 100),
                       intrinsics=LENS, arrival=timestamp + 0.01)


def cam_tracklets(cam_id: int, timestamp: float) -> CameraTracklets:
    tracklet = CameraTracklet(cam_id=cam_id, id=1, status=TrackingStatus.TRACKED, roi=Rect(0.1, 0.2, 0.3, 0.4), age=1, timestamp=timestamp)
    return CameraTracklets(cam_id, timestamp, [tracklet], arrival=timestamp + 0.033)


def record(root: Path, types: list[FrameType], num_cams: int, feed: Callable[[Recorder], None], name: str = '') -> Path:
    """A recording in `root`'s recordings, of what `feed` hands the recorder; its folder."""
    settings = RecorderSettings()
    settings.num_cams = num_cams
    settings.video_frame_types = types
    settings.video_encoder = video.CoderType.iGPU
    settings.temp_path = str(root / 'temp')
    settings.output_path = 'recordings'
    settings.fps = 60.0
    settings.keyframe_interval = 1
    settings.name = name                # two recordings in one second have their own folders
    rec = Recorder(settings, str(root))
    before = set((root / 'recordings').iterdir()) if (root / 'recordings').is_dir() else set()
    rec.start()
    try:
        rec.settings.start = True
        deadline = time.perf_counter() + 5.0
        while rec._get_state() != RecState.REC and time.perf_counter() < deadline:
            time.sleep(0.01)
        feed(rec)
        rec.settings.stop = True
        while rec._get_state() != RecState.IDLE and time.perf_counter() < deadline + 20.0:
            time.sleep(0.01)
    finally:
        rec.stop()
    folders = set((root / 'recordings').iterdir()) - before
    assert len(folders) == 1
    return folders.pop()


def player(root: Path, num_cams: int) -> Player:
    settings = PlayerSettings()
    settings.num_cams = num_cams
    settings.video_frame_types = [FrameType.VIDEO, FrameType.DEPTH]
    settings.video_decoder = video.CoderType.iGPU
    settings.video_path = 'recordings'
    settings.fps = 120.0
    return Player(settings, str(root))


# What a player hands over: (cam, kind, pixel value, recorded capture, capture on the play's clock, metadata)
Played = tuple[int, FrameType, int, float | None, float | None, FrameMetadata | None]


def collect(play: Player, out: list[Played]) -> None:
    play.add_frame_callback(lambda cam, kind, image, frame, captured: out.append(
        (cam, kind, int(np.median(image[..., 0] if image.ndim == 3 else image)), frame.timestamp if frame else None, captured, frame)))


FPS: float = 60.0
SPLIT: int = 30          # frames per camera before the split
COUNT: int = 60
