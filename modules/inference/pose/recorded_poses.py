"""The host's poses recorded with the video, and played back in place of the pose model.

Recording: `frame_poses` turns the RAW stage's poses into what the camera's recorder keeps of them (`FramePoses`,
per camera: each pose's crop box and the model's points with their scores, by track_id), stamped with their
arrival, the one clock read for them: the poses come from the host's pipeline, not from a camera backend. An app
hands it to its recorder, which runs it on its own thread.

Playback: `RecordedPoses` is the RAW stage's async output in -sim, between the crop step and the pose model: it
takes the crop step's poses with their crops (`submit`, an `ImageCallback`) and emits the stage's poses. It hears the
player's frames (`set_clip_frame`, a `ClipFrameCallback`) and keeps each frame's recorded poses by its capture on the
play's clock, the stamp the crops give the stage's poses. With `enabled` on, it sets each cropped pose's box and points
from the recording by track_id, without the pose model, and hands them on after the recorded latency, from its own
thread; a frame the recording has no poses for sends none, and a pose it has no track_id for goes on without points,
counted. With `enabled` off, the crops go to the pose model, as live.
"""

import heapq
import logging
import time
from collections import defaultdict
from threading import Lock, Thread
import numpy as np

from modules.camera import CaptureStream, FrameType, MissedParts
from modules.recording import FrameMetadata, FramePoses, PoseArrays
from modules.pose import BBox, Frame, FrameDict, FrameDictCallbackMixin, Points2D, replace
from modules.settings import BaseSettings, Field
from modules.utils import Rect, Waker

from ..preprocessing import crop
from .pose_estimator import PoseEstimator

logger = logging.getLogger(__name__)

KEPT: int = 32          # recorded frames kept per camera: the stage asks for a frame's poses a few frames after it came


class RecordedPosesSettings(BaseSettings):
    enabled: Field[bool] = Field(False, description="Play the recording's poses in place of the pose model")


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

    def __init__(self, settings: RecordedPosesSettings, pose_estimator: PoseEstimator) -> None:
        super().__init__()
        self.settings: RecordedPosesSettings = settings
        self._pose: PoseEstimator = pose_estimator
        self._pose.add_frames_callback(self._on_model)
        self._lock: Lock = Lock()
        self._recorded: dict[int, CaptureStream[FrameMetadata]] = defaultdict(lambda: CaptureStream(KEPT))
        self.missed: MissedParts = MissedParts('Recorded poses')       # poses whose track_id the recording has not
        self.missed.started()
        # The poses waiting for their moment: (due, order, poses), on this node's thread
        self._waiting: list[tuple[float, int, FrameDict]] = []
        self._order: int = 0
        self._waiting_lock: Lock = Lock()
        self._waker: Waker = Waker()
        self._running: bool = False
        self._thread: Thread | None = None

    def start(self) -> None:
        self._pose.start()
        self._running = True
        self._thread = Thread(target=self._send, daemon=True, name='RecordedPoses')
        self._thread.start()

    def stop(self) -> None:
        with self._waiting_lock:
            self._running = False
            self._waiting.clear()
        self._waker.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._thread is None or not self._thread.is_alive():
            self._waker.close()
        self._pose.stop()

    def reset(self) -> None:
        with self._waiting_lock:
            self._waiting.clear()
        self._pose.reset()

    # ── The player's frames ────────────────────────────────────────────

    def set_clip_frame(self, cam_id: int, frame_type: FrameType, image: np.ndarray, metadata: FrameMetadata | None,
                       captured: float | None) -> None:
        """A `ClipFrameCallback`, on the player's thread: the frame's recorded poses kept by its capture on the
        play's clock."""
        if frame_type != FrameType.VIDEO or metadata is None or captured is None:
            return
        with self._lock:
            self._recorded[cam_id].add(captured, metadata)

    def clear_waiting(self) -> None:
        """The player's jump callback (a loop, a seek): the poses still waiting, and the frames kept, belong to
        the frames before it, and are dropped."""
        with self._lock:
            for recorded in self._recorded.values():
                recorded.clear()
        with self._waiting_lock:
            self._waiting.clear()

    # ── The async step ─────────────────────────────────────────────────

    def submit(self, cropped: FrameDict, crops: crop.CropImageDict) -> None:
        """The crop step's poses with their crops: with `enabled`, the recorded poses played in place of the pose
        model; without, the crops to the pose model, as live."""
        if not self.settings.enabled:
            self._pose.submit(cropped, crops)
            return
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
        with self._waiting_lock:
            heapq.heappush(self._waiting, (due, self._order, played))
            self._order += 1
        self._waker.set()

    def _send(self) -> None:
        """This node's thread: each waiting frame's poses out at their moment."""
        while True:
            with self._waiting_lock:
                if not self._running:
                    return
                due: float | None = self._waiting[0][0] if self._waiting else None
                frames: FrameDict | None = None
                if due is not None and due <= time.perf_counter():
                    _, _, frames = heapq.heappop(self._waiting)
            if frames is None:
                self._waker.wait(due)
                continue
            self._notify_frames_callbacks(frames)
            self.missed.warn()

    def _on_model(self, frames: FrameDict) -> None:
        """The pose model's poses: on to the stage while the recorded ones do not play."""
        if not self.settings.enabled:
            self._notify_frames_callbacks(frames)
