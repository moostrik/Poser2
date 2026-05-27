import time
from threading import Barrier

from ._usb_camera import UsbCamera
from ._usb_camera_player import UsbCameraPlayer
from ._definitions import resolve_device_infos, get_device_list, FrameCallback, SyncCallback, TrackerCallback
from .settings import CameraSettings
from ..player.settings import SimulatorSettings
from ..player import Player

import logging
logger = logging.getLogger(__name__)


class UsbCameraPlayers:
    """Group of UsbCameraPlayer instances — simulates USB camera input from video files.

    Mirrors the UsbCameras interface so apps can use either interchangeably
    via the Cameras protocol.
    """

    def __init__(self, player: Player, settings_list: list[CameraSettings], player_settings: SimulatorSettings) -> None:
        n = len(settings_list)
        barrier = Barrier(n)
        infos = resolve_device_infos([s.device_id for s in settings_list])
        # Only non-passthrough players open a real device and trigger a USB reset on stop.
        self._opened_ids: set[str] = (
            {s.device_id for s in settings_list if infos[s.device_id] is not None}
            if not player_settings.sim_passthrough else set()
        )
        self._cameras: list[UsbCameraPlayer] = [
            UsbCameraPlayer(player, s, player_settings, barrier, infos[s.device_id])
            for s in settings_list
        ]

    # --- Lifecycle ---

    def start(self) -> None:
        for camera in self._cameras:
            camera.start()

    def stop(self) -> None:
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
