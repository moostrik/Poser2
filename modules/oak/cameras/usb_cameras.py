import time
from threading import Barrier

from ._usb_camera import UsbCamera
from ._definitions import resolve_device_infos, get_device_list, FrameCallback, SyncCallback, TrackerCallback
from .settings import CameraSettings

import logging
logger = logging.getLogger(__name__)


class UsbCameras:
    """USB-coordinated group of OAK cameras.

    Cameras on a shared USB hub are not independent — their enumeration, start,
    and stop phases are coupled at the bus level. This class owns the per-batch
    Barrier and resolves device info on the main thread before any camera thread
    starts, so each UsbCamera worker receives a stable, consistent view of the bus
    state.

    Works for n=1: Barrier(1) is trivially satisfied (no waiting).
    """

    def __init__(self, settings_list: list[CameraSettings]) -> None:
        UsbCamera._id_counter = 0
        n = len(settings_list)
        barrier = Barrier(n)
        infos = resolve_device_infos([s.device_id for s in settings_list])
        self._opened_ids: set[str] = {
            s.device_id for s in settings_list if infos[s.device_id] is not None
        }
        self._cameras: list[UsbCamera] = [
            UsbCamera(s, barrier, infos[s.device_id])
            for s in settings_list
        ]

    # --- Lifecycle ---

    def start(self) -> None:
        for camera in self._cameras:
            camera.start()

    def stop(self) -> None:
        """Blocking stop: signal all cameras, join all threads, then wait for USB re-enumeration.

        After pipeline.stop() each device resets on USB and takes 1–3s to reappear.
        Polling here ensures the next app run sees all devices as AVAILABLE.
        """
        for camera in self._cameras:
            camera.stop()
        for camera in self._cameras:
            camera.join(timeout=10.0)

        if not self._opened_ids:
            return

        max_tries = 6
        missing: set[str] = set()
        for attempt in range(1, max_tries + 1):
            available = set(get_device_list())
            missing = self._opened_ids - available
            if not missing:
                logger.info(f'All cameras back on USB bus — safe to restart (attempt {attempt}).')
                return
            logger.warning(f'Cameras still resetting on USB bus (attempt {attempt}/{max_tries}): {missing}')
            time.sleep(0.5)
        logger.error(f'USB reset timed out — cameras may not be available on next run: {missing}')

    # --- Access ---

    def get(self, device_id: str) -> UsbCamera | None:
        for camera in self._cameras:
            if camera.device_id == device_id:
                return camera
        return None

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
