from threading import Barrier, BrokenBarrierError
from typing import Iterator

from ._camera import Camera
from ._definitions import get_device_info, FrameCallback, SyncCallback, TrackerCallback
from .settings import CameraSettings

import logging
logger = logging.getLogger(__name__)


class Cameras:
    """USB-coordinated group of OAK cameras.

    Cameras on a shared USB hub are not independent — their enumeration, start,
    and stop phases are coupled at the bus level. This class owns the per-batch
    Barrier and resolves device info on the main thread before any camera thread
    starts, so each _Camera worker receives a stable, consistent view of the bus
    state.

    Works for n=1: Barrier(1) is trivially satisfied (no waiting).
    """

    def __init__(self, settings_list: list[CameraSettings]) -> None:
        n = len(settings_list)
        barrier = Barrier(n)
        device_infos = [get_device_info(s.device_id) for s in settings_list]
        self._cameras: list[Camera] = [
            Camera(s, barrier, info)
            for s, info in zip(settings_list, device_infos)
        ]

    # --- Iteration / access ---

    def __iter__(self) -> Iterator[Camera]:
        return iter(self._cameras)

    def __len__(self) -> int:
        return len(self._cameras)

    def __getitem__(self, index: int) -> Camera:
        return self._cameras[index]

    # --- Lifecycle ---

    def start(self) -> None:
        for camera in self._cameras:
            camera.start()

    def stop(self) -> None:
        for camera in self._cameras:
            camera.stop()

    def join(self, timeout: float = 10.0) -> None:
        for camera in self._cameras:
            camera.join(timeout=timeout)

    # --- Callbacks ---

    def add_frame_callback(self, cb: FrameCallback) -> None:
        for camera in self._cameras:
            camera.add_frame_callback(cb)

    def add_sync_callback(self, cb: SyncCallback) -> None:
        for camera in self._cameras:
            camera.add_sync_callback(cb)

    def add_tracker_callback(self, cb: TrackerCallback) -> None:
        for camera in self._cameras:
            camera.add_tracker_callback(cb)

    def add_preview_callback(self, cb: FrameCallback) -> None:
        for camera in self._cameras:
            camera.add_preview_callback(cb)
