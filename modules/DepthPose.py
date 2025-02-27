
from modules.cam.DepthAiGui import DepthAiGui as DepthCam
from modules.render.Render import Render
from modules.gui.PyReallySimpleGui import Gui

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

        self.gui: Gui = Gui('DepthPose', path + '/files/', 'default')
        self.render: Render = Render(self.width, self.height , self.width, self.height * 2, 'Depth Pose', fullscreen=False, v_sync=True, stretch=False)

        self.camera = DepthCam(self.gui, (self.width, self.height), True)
        self.camera.setRotate90(self.portrait_mode)

        self._running: bool = False


    def start(self) -> None:
        self.camera.open()
        self.camera.startCapture()
        self.camera.addColorCallback(self.render.set_video_image)
        self.camera.addStereoCallback(self.render.set_depth_image)

        self.render.exit_callback = self.stop
        self.render.addKeyboardCallback(self.render_keyboard_callback)
        self.render.start()

        self.gui.exit_callback = self.stop
        self.gui.addFrame([self.camera.get_color_frame(), self.camera.get_stereo_frame()])
        self.gui.start()

        self._running = True

    def stop(self) -> None:
        self.camera.stopCapture()
        self.camera.clearColorCallbacks()
        self.camera.close()

        self.render.exit_callback = None
        self.render.stop()

        self.gui.exit_callback = None
        self.gui.stop()

        self._running = False

    def isRunning(self) -> bool :
        return self._running

    def render_keyboard_callback(self, key, x, y) -> None:
        if key == b' ': # space
            pass