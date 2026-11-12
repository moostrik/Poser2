"""The host's poses recorded with the video, and played back in place of the pose model.

Recording: `frame_poses` turns the RAW stage's poses into what the camera's recorder keeps of them (`FramePoses`,
per camera: each pose's crop box and the model's points with their scores, by track_id), stamped with their
arrival here, the one clock read for them: the poses come from the host's pipeline, not from a camera backend.

Playback: `RecordedPoses` is the RAW stage's async step in -sim, around `PoseInference`. It hears the player's
frames (`set_clip_frame`, a `ClipFrameCallback`) and keeps each frame's recorded poses by its capture on the play's
clock, the stamp the crops give the stage's poses. With `enabled` on, it cuts the crops without the pose model
(`PoseInference.cut_crops`), sets each pose's box and points from the recording by track_id, and hands them on after the
recorded latency, from its own thread; a frame the recording has no poses for sends none, and a pose it has no
track_id for goes on without points, counted. With `enabled` off, the pose model runs as live.
"""

import heapq
import logging
import time
from collections import defaultdict
from threading import Condition, Lock, Thread

import numpy as np

from modules.camera import CaptureStream, FrameMetadata, FramePoses, FrameType, MissedParts, PoseArrays
from modules.pose import BBox, Frame, FrameDict, FrameDictCallbackMixin, Points2D, replace
from modules.settings import BaseSettings, Field
from modules.utils import Rect

from .pose_inference import PoseInference

logger = logging.getLogger(__name__)

KEPT: int = 32          # recorded frames kept per camera: the stage asks for a frame's poses a few frames after it came
JUMP: float = 1.0       # s of recorded time between two frames beyond which the play jumped: what waits is dropped


class RecordedPosesSettings(BaseSettings):
    enabled: Field[bool] = Field(True, description="Play the recording's poses in place of the pose model")


def pose_arrays(frames: FrameDict) -> dict[int, tuple[float, PoseArrays]]:
    """The poses with points, per camera: their frame's capture time and each pose's box and points."""
    by_cam: dict[int, tuple[float, PoseArrays]] = {}
    for track_id, frame in frames.items():
        points: Points2D = frame[Points2D]
        if not np.any(points.scores > 0.0):
            continue
        rect: Rect = frame[BBox].to_rect()
        box: np.ndarray = np.array([rect.x, rect.y, rect.width, rect.height], dtype=np.float32)
        arrays: np.ndarray = np.column_stack((points.values, points.scores)).astype(np.float32)
        timestamp, cam_poses = by_cam.get(frame.cam_id, (frame.timestamp, {}))
        cam_poses[track_id] = (box, arrays)
        by_cam[frame.cam_id] = (max(timestamp, frame.timestamp), cam_poses)
    return by_cam


def frame_poses(frames: FrameDict, arrival: float) -> list[FramePoses]:
    """The RAW stage's poses as the recorder keeps them, one per camera, handed over at `arrival`."""
    return [FramePoses(cam_id, timestamp, poses, arrival) for cam_id, (timestamp, poses) in pose_arrays(frames).items()]


def recorded_pose(frame: Frame, box: np.ndarray, points: np.ndarray) -> Frame:
    """`frame` with the recorded box and points in place of its own."""
    return replace(frame, {BBox: BBox.from_rect(Rect(*(float(v) for v in box))),
                           Points2D: Points2D(values=points[:, :2].copy(), scores=points[:, 2].copy())})


class RecordedPoses(FrameDictCallbackMixin):
    """See the module docstring."""

    def __init__(self, settings: RecordedPosesSettings, pose_inference: PoseInference) -> None:
        super().__init__()
        self.settings: RecordedPosesSettings = settings
        self._inference: PoseInference = pose_inference
        self._inference.add_frames_callback(self._on_model)
        self._lock: Lock = Lock()
        self._recorded: dict[int, CaptureStream[FrameMetadata]] = defaultdict(lambda: CaptureStream(KEPT))
        self._recorded_time: dict[int, float] = {}     # each camera's last recorded capture time
        self.missed: MissedParts = MissedParts('Recorded poses')       # poses whose track_id the recording has not
        self.missed.started()
        # The poses waiting for their moment: (due, order, poses), on this node's thread
        self._waiting: list[tuple[float, int, FrameDict]] = []
        self._order: int = 0
        self._wake: Condition = Condition()
        self._running: bool = False
        self._thread: Thread | None = None

    def start(self) -> None:
        self._inference.start()
        self._running = True
        self._thread = Thread(target=self._send, daemon=True, name='RecordedPoses')
        self._thread.start()

    def stop(self) -> None:
        with self._wake:
            self._running = False
            self._waiting.clear()
            self._wake.notify()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._inference.stop()

    def reset(self) -> None:
        with self._wake:
            self._waiting.clear()
        self._inference.reset()

    # ── The player's frames ────────────────────────────────────────────

    def set_clip_frame(self, cam_id: int, frame_type: FrameType, image: np.ndarray, metadata: FrameMetadata | None,
                       captured: float | None) -> None:
        """A `ClipFrameCallback`, on the player's thread: the frame's recorded poses kept by its capture on the
        play's clock. A jump in the recorded time drops the poses still waiting."""
        if frame_type != FrameType.VIDEO or metadata is None or captured is None:
            return
        with self._lock:
            last: float | None = self._recorded_time.get(cam_id)
            jumped: bool = last is not None and not 0.0 < metadata.timestamp - last <= JUMP
            self._recorded_time[cam_id] = metadata.timestamp
            if jumped:
                self._recorded[cam_id].clear()
            self._recorded[cam_id].add(captured, metadata)
        if jumped:
            with self._wake:
                self._waiting.clear()

    # ── The async step ─────────────────────────────────────────────────

    def submit(self, frames: FrameDict) -> None:
        """The stage's poses: with `enabled`, the crops cut without the pose model and the recorded poses played;
        without, the crops and the pose model, as live."""
        if not self.settings.enabled:
            self._inference.submit(frames)
            return
        cropped, _ = self._inference.cut_crops(frames)
        by_cam: dict[int, FrameDict] = defaultdict(dict)
        for track_id, frame in cropped.items():
            by_cam[frame.cam_id][track_id] = frame
        for cam_id, cam_frames in by_cam.items():
            self._play(cam_id, cam_frames)

    def _play(self, cam_id: int, frames: FrameDict) -> None:
        stamp: float = max(frame.timestamp for frame in frames.values())     # the image's capture, the crops' stamp
        with self._lock:
            metadata: FrameMetadata | None = self._recorded[cam_id].at(stamp)
        if metadata is None or metadata.poses is None:
            return
        recorded: PoseArrays = metadata.poses
        played: FrameDict = {}
        for track_id, frame in frames.items():
            if track_id in recorded:
                played[track_id] = recorded_pose(frame, *recorded[track_id])
            else:
                played[track_id] = frame
                self.missed.count(cam_id, 'track_id')
        due: float = stamp + (metadata.pose_latency or 0.0)
        with self._wake:
            heapq.heappush(self._waiting, (due, self._order, played))
            self._order += 1
            self._wake.notify()

    def _send(self) -> None:
        """This node's thread: each waiting frame's poses out at their moment."""
        while True:
            with self._wake:
                while self._running and (not self._waiting or self._waiting[0][0] > time.perf_counter()):
                    self._wake.wait(self._waiting[0][0] - time.perf_counter() if self._waiting else None)
                if not self._running:
                    return
                _, _, frames = heapq.heappop(self._waiting)
            self._notify_frames_callbacks(frames)
            self.missed.warn()

    def _on_model(self, frames: FrameDict) -> None:
        """The pose model's poses: on to the stage while the recorded ones do not play."""
        if not self.settings.enabled:
            self._notify_frames_callbacks(frames)
