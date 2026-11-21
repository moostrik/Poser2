"""The cameras' sync: the cameras' messages of one moment, one per camera, as a set.

The cameras run at one frame rate, so each keeps its phase, its place in the frame cycle (its capture time modulo
`1 / fps`, averaged over `PHASE_FRAMES` frames). The rounds are cut in the middle of the largest gap between the
phases of the live cameras (those with a message within `LIVE_FRAMES` frames). A set closes as soon as every live camera is in it, or when a message of a later round, or a
camera's second message, comes; a camera that did not come is then left out. Every message is in one set.

`submit` only queues a message, on the thread it comes on, at most `WAITING` per camera; beyond that the oldest
is dropped, counted in `dropped`. The sync's own thread decides the sets in the order the messages came and hands
each, by `cam_id`, to the set callbacks outside its lock. A submit before `start` is an error; one after `stop` is
dropped."""

import logging
import math
import time
from collections import deque
from threading import Condition, Lock, Thread
from typing import Callable, Generic, TypeVar

from modules.camera import CameraFrame, CameraTracklets, FrameType
from modules.settings import BaseSettings, Field
from modules.utils import PerformanceTimer

logger = logging.getLogger(__name__)

WAITING: int = 16       # messages per camera waiting for the sync's thread
PHASE_FRAMES: int = 10  # frames a camera's phase is averaged over
LIVE_FRAMES: int = 2    # a camera without a message for this many frames is left out of the sets

SyncMessage = CameraFrame | CameraTracklets
M = TypeVar('M', CameraFrame, CameraTracklets)   # the message one sync pairs
SyncSetCallback = Callable[[dict[int, M]], None]


class CameraSyncSettings(BaseSettings):
    fps: Field[float] = Field(30.0, access=Field.INIT, visible=False)


class CameraSync(Generic[M]):
    """The cameras' sync, see the module docstring. Frames and tracklet messages are two separate instances with the
    same rule (`frame_sync`, `tracklet_sync`)."""

    def __init__(self, settings: CameraSyncSettings, stream_name: str = '') -> None:
        self.name: str = f'CameraSync {stream_name}'.strip()
        self._thread: Thread = Thread(target=self._run, daemon=True, name=self.name)
        self.stream_name: str = stream_name
        self._period: float = 1.0 / settings.fps

        # On the deciding thread
        self._phases: dict[int, tuple[float, float]] = {}  # each camera's phase, as an averaged unit vector
        self._newest: dict[int, float] = {}                 # each camera's newest capture time
        self._open: dict[int, M] = {}                       # the set being filled
        self._open_round: int | None = None

        self._callback_lock: Lock = Lock()
        self._callbacks: list[SyncSetCallback[M]] = []      # run in registration order, with the set
        self._wake: Condition = Condition()
        self._running: bool = False
        self._inbox: deque[tuple[M, float]] = deque()               # (message, the moment it was queued)
        self._queued: dict[int, int] = {}                           # what the inbox holds per camera
        self.dropped: int = 0
        self._stopped: bool = False
        self._wait: PerformanceTimer = PerformanceTimer(name=f"Wait camera sync {stream_name}".strip())

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        with self._wake:
            self._running = True
        self._thread.start()

    def stop(self) -> None:
        """The thread ends; the messages still waiting are dropped and logged."""
        with self._wake:
            self._running = False
            self._stopped = True
            dropped: int = len(self._inbox)
            self._inbox.clear()
            self._queued.clear()
            self._wake.notify()
        if dropped:
            logger.info("%s: %d messages dropped at stop", self.name, dropped)
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._wake:
                while self._running and not self._inbox:
                    self._wake.wait()
                if not self._running:
                    return
                message, queued = self._inbox.popleft()
                self._queued[message.cam_id] -= 1
            self._wait.add_time((time.perf_counter() - queued) * 1000.0, report=False)
            try:
                self._decide(message)
            except Exception:
                logger.exception("%s: a sync failed", self.name)

    # ── Inputs: each queues and returns ───────────────────────────────

    def submit(self, message: M) -> None:
        """A camera's frame or tracklet callback: the message, queued for the thread, the camera's oldest dropped beyond
        `WAITING`. Of the frames only VIDEO counts: a sync is one frame per camera."""
        if isinstance(message, CameraFrame) and message.kind != FrameType.VIDEO:
            return
        cam_id: int = message.cam_id
        with self._wake:
            if not self._running:
                if self._stopped:
                    return
                raise RuntimeError(f"{self.name}: a message before start")
            if self._queued.get(cam_id, 0) >= WAITING:
                for index, (queued_message, _) in enumerate(self._inbox):
                    if queued_message.cam_id == cam_id:
                        del self._inbox[index]
                        self.dropped += 1
                        break
            else:
                self._queued[cam_id] = self._queued.get(cam_id, 0) + 1
            self._inbox.append((message, time.perf_counter()))
            self._wake.notify()

    def _decide(self, message: M) -> None:
        """The message into its round's set; the set goes out once complete, or once a later round has begun."""
        cam_id: int = message.cam_id
        timestamp: float = message.timestamp
        period: float = self._period
        angle: float = 2.0 * math.pi * (timestamp % period) / period
        x, y = self._phases.get(cam_id, (math.cos(angle), math.sin(angle)))
        self._phases[cam_id] = (x + (math.cos(angle) - x) / PHASE_FRAMES, y + (math.sin(angle) - y) / PHASE_FRAMES)
        self._newest[cam_id] = timestamp
        live: set[int] = {cam for cam, newest in self._newest.items() if timestamp - newest <= LIVE_FRAMES * period}

        # The round of the camera's own frame cycle the message is nearest to, so its jitter never moves it across the cut
        x, y = self._phases[cam_id]
        phase: float = (math.atan2(y, x) % (2.0 * math.pi)) / (2.0 * math.pi) * period
        expected: float = phase + round((timestamp - phase) / period) * period
        round_index: int = math.floor((expected - self._cut(live)) / period)
        if self._open_round is not None and (round_index > self._open_round or cam_id in self._open):
            self._close()
        if self._open_round is None:
            self._open_round = round_index
        self._open[cam_id] = message
        if live <= self._open.keys():
            self._close()

    def _cut(self, live: set[int]) -> float:
        """Where the rounds are cut in the frame cycle (s): the middle of the largest gap between the live cameras'
        phases."""
        period: float = self._period
        phases: list[float] = sorted((math.atan2(self._phases[cam][1], self._phases[cam][0]) % (2.0 * math.pi))
                                     / (2.0 * math.pi) * period for cam in live)
        width, start = max(((phases[(i + 1) % len(phases)] - phase) % period or period, phase)
                           for i, phase in enumerate(phases))
        return (start + width / 2.0) % period

    def _close(self) -> None:
        items: dict[int, M] = self._open
        self._open = {}
        self._open_round = None
        self._notify_callbacks(items)

    def add_set_callback(self, callback: SyncSetCallback[M]) -> None:
        """Every set, one message per camera by `cam_id`, on the deciding thread."""
        with self._callback_lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def _notify_callbacks(self, items: dict[int, M]) -> None:
        with self._callback_lock:
            callbacks: list[SyncSetCallback[M]] = list(self._callbacks)
        for callback in callbacks:
            try:
                callback(dict(items))
            except Exception:
                logger.exception("%s: error in a sync callback", self.name)
