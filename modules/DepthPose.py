from modules.cam.DepthCam import DepthCam, DepthSimulator
from modules.cam.recorder.SyncRecorderGui import SyncRecorderGui as Recorder, EncoderType
from modules.cam.depthplayer.SyncPlayerGui import SyncPlayerGui as Player, DecoderType
from modules.render.Render import Render
from modules.gui.PyReallySimpleGui import Gui
from modules.person.pose.PoseDetection import ModelType
from modules.person.Manager import Manager

from modules.cam.depthcam.Definitions import FrameType
from modules.cam.depthcam.Pipeline import get_frame_types

import os
from enum import Enum

class CamType(Enum):
    DEPTH   = 1
    VIMBA   = 2
    WEB     = 3
    IMAGE   = 4

class DepthPose():
    def __init__(self, path: str, camera_list: list[str], fps: int, numPlayers: int,
                 color: bool, stereo: bool, person: bool, lowres: bool, showStereo: bool,
                 lightning: bool, noPose:bool, simulation: bool) -> None:
        self.path: str =    path
        modelPath: str =    os.path.join(path, 'models')
        recorderPath: str = os.path.join(path, 'recordings')
        self.noPose: bool = noPose

        frame_types: list[FrameType] = get_frame_types(color, stereo, showStereo)
        num_cameras: int = len(camera_list)
        print(f'num_cameras: {num_cameras}')

        self.gui = Gui('DepthPose', os.path.join(path, 'files'), 'default')
        self.render = Render(num_cameras, numPlayers, 1280, 720 + 256, 'Depth Pose', fullscreen=False, v_sync=True)

        self.recorder = Recorder(self.gui, recorderPath, num_cameras, frame_types, 10.0, EncoderType.iGPU)
        self.player: Player = Player(recorderPath, num_cameras, frame_types, DecoderType.CPU)

        # DepthCam.get_device_list(verbose=True)

        self.cameras: list[DepthCam | DepthSimulator] = []
        if simulation:
            for cam_id in camera_list:
                self.cameras.append(DepthSimulator(self.gui, self.player, cam_id, modelPath, fps, color, stereo, person, lowres, showStereo))
        else:
            for cam_id in camera_list:
                camera = DepthCam(self.gui, cam_id, modelPath, fps, color, stereo, person, lowres, showStereo)
                self.cameras.append(camera)

        if len(self.cameras) == 0:
            print('No cameras available')

        modelType: ModelType = ModelType.LIGHTNING if lightning else ModelType.THUNDER
        if self.noPose:
            modelType = ModelType.NONE

        self.detector = Manager(max_persons=numPlayers, num_cams=1, model_path=modelPath, model_type=modelType)

        self.running: bool = False


    def start(self) -> None:
        self.render.exit_callback = self.stop
        self.render.addKeyboardCallback(self.render_keyboard_callback)
        self.render.start()

        for camera in self.cameras:
            camera.start()
            camera.add_preview_callback(self.detector.set_image)
            camera.add_preview_callback(self.render.set_cam_image)
            camera.add_tracker_callback(self.detector.add_tracklet)
            camera.add_tracker_callback(self.render.add_tracklet)
            for T in self.recorder.types:
                camera.add_frame_callback(T, self.recorder.add_frame)
                camera.add_fps_callback(self.recorder.set_fps)

        self.detector.start()
        self.detector.addCallback(self.render.add_person)

        self.recorder.start()
        self.player.start()

        self.gui.exit_callback = self.stop

        for camera in self.cameras:
            self.gui.addFrame([camera.gui.get_gui_color_frame(), camera.gui.get_gui_depth_frame()])
        self.gui.addFrame([self.recorder.get_gui_frame(), self.player.get_gui_frame()])
        self.gui.start()
        self.gui.bringToFront()

        for camera in self.cameras:
            camera.gui.gui_check()
        self.recorder.gui_check() # start after gui to prevent record at startup

        self.running = True

    def stop(self) -> None:
        for camera in self.cameras:
            camera.stop()

        self.player.stop()
        self.detector.stop()
        self.recorder.stop()

        self.gui.exit_callback = None
        self.gui.stop()

        self.detector.join()
        self.recorder.join()
        self.player.join()
        for camera in self.cameras:
            camera.join()


        self.render.exit_callback = None
        self.render.stop()
        # self.render.join()

        self.running = False

    def isRunning(self) -> bool :
        return self.running

    def render_keyboard_callback(self, key, x, y) -> None:
        if not  self.isRunning(): return
        if key == b'g' or key == b'G':
            if not self.gui or not self.gui.isRunning(): return
            self.gui.bringToFront()
