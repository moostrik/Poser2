from ._usb_camera import UsbCamera
from ._usb_camera_player import UsbCameraPlayer
from ._definitions import FrameCallback, SyncCallback, TrackerCallback
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
        self._cameras: list[UsbCameraPlayer] = [
            UsbCameraPlayer(player, s, player_settings)
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
