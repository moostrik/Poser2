# Standard library imports
from math import ceil
from typing import Optional

# Local application imports
from modules.av.Manager import Manager as AV
from modules.cam.DepthCam import DepthCam, DepthSimulator
from modules.cam.recorder.SyncRecorderGui import SyncRecorderGui as Recorder
from modules.cam.depthplayer.SyncPlayerGui import SyncPlayerGui as Player
from modules.gui.PyReallySimpleGui import Gui
from modules.tracker.panoramic.PanoramicTracker import PanoramicTracker as PanoramicTracker
from modules.correlation.DTWCorrelator import DTWCorrelator
from modules.pose.PosePipeline import PosePipeline
from modules.pose.PoseStreamProcessor import PoseStreamProcessor
from modules.correlation.PairCorrelationStream import PairCorrelationStreamProcessor
from modules.render.Render import Render
from modules.Settings import Settings


class Main():
    def __init__(self, settings: Settings) -> None:
        self.gui = Gui(settings.render_title + ' GUI', settings.path_file, 'default')

        self.settings: Settings = settings

        self.render = Render(settings)

        self.cameras: list[DepthCam | DepthSimulator] = []
        self.recorder: Optional[Recorder] = None
        self.player: Optional[Player] = None
        if settings.camera_simulation:
            self.player = Player(self.gui, settings)
            for cam_id in settings.camera_list:
                self.cameras.append(DepthSimulator(self.gui, self.player, cam_id, settings))
        else:
            self.recorder = Recorder(self.gui, settings)
            for cam_id in settings.camera_list:
                camera = DepthCam(self.gui, cam_id, settings)
                self.cameras.append(camera)

        self.panoramic_tracker = PanoramicTracker(self.gui, settings)

        self.pose_detection = PosePipeline(settings)
        self.pose_streamer = PoseStreamProcessor(settings)
        self.dtw_correlator = DTWCorrelator(settings)
        self.correlation_streamer = PairCorrelationStreamProcessor(settings)

        self.av: AV = AV(self.gui, settings)
        self.running: bool = False

    def start(self) -> None:
        self.render.exit_callback = self.stop
        self.render.addKeyboardCallback(self.render_keyboard_callback)
        self.render.start()

        for camera in self.cameras:
            camera.add_preview_callback(self.render.set_cam_image)
            if self.recorder:
                camera.add_sync_callback(self.recorder.set_synced_frames)
            camera.add_frame_callback(self.pose_detection.set_image)
            camera.add_tracker_callback(self.panoramic_tracker.add_cam_tracklet)
            camera.add_tracker_callback(self.render.set_depth_tracklet)
            camera.start()

        self.correlation_streamer.add_stream_callback(self.render.set_correlation_stream)
        self.correlation_streamer.start()

        self.dtw_correlator.add_correlation_callback(self.correlation_streamer.add_correlation)
        self.dtw_correlator.start()

        # self.pose_streamer.add_stream_callback(self.dtw_correlator.set_pose_stream)
        self.pose_streamer.add_stream_callback(self.render.set_pose_stream)
        self.pose_streamer.start()

        # self.pose_detection.add_pose_callback(self.pose_streamer.add_pose)
        self.pose_detection.add_pose_callback(self.render.set_pose)
        self.pose_detection.start()

        self.panoramic_tracker.add_tracklet_callback(self.pose_detection.add_tracklet)
        self.panoramic_tracker.add_tracklet_callback(self.render.set_tracklet)
        self.panoramic_tracker.start()

        self.av.add_output_callback(self.render.set_av)
        self.av.start()

        # GUIGUIGUIGUIGUIGUIGUIGUIGUIGUIGUIGUI
        self.gui.exit_callback = self.stop

        for i in range(ceil(len(self.cameras) / 2.0)):
            c: int = i * 2
            if c + 1 < len(self.cameras):
                self.gui.addFrame([self.cameras[c].gui.get_gui_frame(), self.cameras[c+1].gui.get_gui_frame()])
            else:
                self.gui.addFrame([self.cameras[c].gui.get_gui_frame()])
        self.gui.addFrame([self.av.gui.get_gui_frame()])

        if self.player:
            self.gui.addFrame([self.player.get_gui_frame(), self.panoramic_tracker.gui.get_gui_frame()])
        if self.recorder:
            self.gui.addFrame([self.recorder.get_gui_frame(), self.panoramic_tracker.gui.get_gui_frame()])
        self.gui.start()
        self.gui.bringToFront()
        # GUIGUIGUIGUIGUIGUIGUIGUIGUIGUIGUIGUI

        for camera in self.cameras:
            camera.gui.gui_check()

        if self.player:
            self.player.gui_check()
            self.player.start()
        if self.recorder:
            self.recorder.gui_check()
            self.recorder.start() # start after gui to prevent record at startup

        self.running = True

    def stop(self) -> None:

        if self.player:
            # print('stop and join player')
            self.player.stop()
            self.player.join()

        # print('stop cameras')
        for camera in self.cameras:
            camera.stop()

        # print('stop detector')
        self.panoramic_tracker.stop()
        self.panoramic_tracker.join()

        if self.pose_detection:
            self.pose_detection.stop()
        if self.pose_streamer:
            self.pose_streamer.stop()
        if self.dtw_correlator:
            self.dtw_correlator.stop()
            self.dtw_correlator.join()
        if self.correlation_streamer:
            self.correlation_streamer.stop()

        self.av.stop()
        self.av.join()

        if self.recorder:
            # print('stop recorder')
            self.recorder.stop()
            self.recorder.join()

        for camera in self.cameras:
            camera.join(timeout=8)

        # print ('stop gui')
        self.gui.stop()
        # self.gui.join() # does not work as stop can be called from gui's own thread

        # print ('stop render')
        self.render.exit_callback = None
        self.render.stop()
        # self.render.join() # does not work as stop can be called from render's own thread

        self.running = False

    def isRunning(self) -> bool :
        return self.running

    def render_keyboard_callback(self, key, x, y) -> None:
        if not  self.isRunning(): return
        if key == b'g' or key == b'G':
            if not self.gui or not self.gui.isRunning(): return
            self.gui.bringToFront()
