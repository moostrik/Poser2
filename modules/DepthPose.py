
from modules.cam.DepthAiGui import DepthAiGui as DepthCam
from modules.render.Render import Render
from modules.gui.PyReallySimpleGui import Gui
from modules.pose.PoseDetection import PoseDetection, ModelType, PoseMessage
import os
from enum import Enum

class CamType(Enum):
    DEPTH   = 1
    VIMBA   = 2
    WEB     = 3
    IMAGE   = 4

class DepthPose():
    def __init__(self, path: str, fps: int, mono: bool, lowres: bool, queueLeft: bool, nopose:bool) -> None:
        self.path: str =    path
        self.noPose: bool = nopose
        self.width: int =   1280
        self.height: int =  720
        if lowres:
            self.width =    640
            self.height =   360

        self.gui = Gui('DepthPose', path + '/files/', 'default')
        self.render = Render(self.width, self.height , self.width, self.height, 'Depth Pose', fullscreen=False, v_sync=True, stretch=False)
        self.camera = DepthCam(self.gui, fps, mono, lowres, queueLeft)
        self.detector = PoseDetection(os.path.join(path, 'models'), ModelType.THUNDER)

        self._running: bool = False

    def start(self) -> None:
        self.render.exit_callback = self.stop
        self.render.addKeyboardCallback(self.render_keyboard_callback)
        self.render.start()

        self.camera.open()
        self.camera.startCapture()
        self.camera.addFrameCallback(self.detector.set_image)

        if not self.noPose:
            self.detector.start()
            self.detector.addMessageCallback(self.pose_callback)
        else:
            self.camera.addFrameCallback(self.render.set_video_image)

        self.gui.exit_callback = self.stop
        self.gui.addFrame([self.camera.get_gui_color_frame(), self.camera.get_gui_depth_frame()])
        self.gui.start()
        self.gui.bringToFront()

        self._running = True

    def stop(self) -> None:
        if not self.noPose:
            self.detector.stop()
            self.detector.clearMessageCallbacks()

        self.camera.stopCapture()
        self.camera.clearFrameCallbacks()
        self.camera.close()

        self.render.exit_callback = None
        self.render.stop()

        self.gui.exit_callback = None
        self.gui.stop()

        self._running = False

    def isRunning(self) -> bool :
        return self._running

    def render_keyboard_callback(self, key, x, y) -> None:
        if not  self.isRunning(): return
        if key == b'g' or key == b'G':
            if not self.gui or not self.gui.isRunning(): return
            self.gui.bringToFront()

    def pose_callback(self, pose_message: PoseMessage) -> None:
        if not  self.isRunning(): return
        self.render.set_video_image(pose_message.image)