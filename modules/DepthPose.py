
from modules.cam.DepthAi import DepthAi as DepthCam
from modules.render.Render import Render

from enum import Enum
import numpy as np

class CamType(Enum):
    DEPTH   = 1
    VIMBA   = 2
    WEB     = 3
    IMAGE   = 4

class DepthPose():
    def __init__(self, path: str, width: int, height: int, portrait_mode: bool) -> None:
        self.path: str = path
        self.width: int = width
        self.height: int = height
        self.portrait_mode: bool = portrait_mode
        if self.portrait_mode:
            self.width: int = height
            self.height: int = width

        self.camera = DepthCam((self.width, self.height))
        self.camera.setRotate90(self.portrait_mode)

        self.render: Render = Render(self.width, self.height , 960, 1080, 'CamTest', fullscreen=False, v_sync=True, stretch=False)

        self._running: bool = False


    def start(self) -> None:
        self.camera.open()
        self.camera.startCapture()
        self.camera.setColorCallback(self.render.set_cam_image)

        self.render.exit_callback = self.stop
        self.render.addKeyboardCallback(self.render_keyboard_callback)
        self.render.start()

        self._running = True

    def stop(self) -> None:
        self.camera.stopCapture()
        self.camera.clearColorCallbacks()
        self.camera.close()

        self.render.exit_callback = None
        self.render.stop()

        self._running = False

    def isRunning(self) -> bool :
        return self._running

    def render_keyboard_callback(self, key, x, y) -> None:
        if key == b' ': # space
            pass