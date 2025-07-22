# Standard library imports
from math import ceil
from typing import Optional

# Local application imports
from modules.av.Manager import Manager as AV
from modules.cam.DepthCam import DepthCam, DepthSimulator
from modules.cam.recorder.SyncRecorderGui import SyncRecorderGui as Recorder
from modules.cam.depthplayer.SyncPlayerGui import SyncPlayerGui as Player
from modules.gui.PyReallySimpleGui import Gui
from modules.tracker.TrackerBase import TrackerType
from modules.tracker.panoramic.PanoramicTracker import PanoramicTracker
from modules.tracker.onepercam.OnePerCamTracker import OnePerCamTracker
from modules.correlation.DTWCorrelator import DTWCorrelator
from modules.pose.PosePipeline import PosePipeline
from modules.pose.PoseStream import PoseStreamManager
from modules.correlation.PairCorrelationStream import PairCorrelationStreamManager
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

        if settings.tracker_type == TrackerType.PANORAMIC:
            self.tracker = PanoramicTracker(self.gui, settings)
        else:
            self.tracker = OnePerCamTracker(self.gui, settings)

        self.pose_detection = PosePipeline(settings)
        self.pose_streamer = PoseStreamManager(settings)
        self.dtw_correlator = DTWCorrelator(settings)
        self.correlation_streamer = PairCorrelationStreamManager(settings)

        self.av = None
        if settings.art_type == Settings.ArtType.WS:
            self.av: AV | None = AV(self.gui, settings)

        self.is_running: bool = False
        self.is_finished: bool = False

    def start(self) -> None:

        for camera in self.cameras:
            camera.add_preview_callback(self.render.data.set_cam_image)
            if self.recorder:
                camera.add_sync_callback(self.recorder.set_synced_frames)
            camera.add_frame_callback(self.pose_detection.set_image)
            camera.add_tracker_callback(self.tracker.add_cam_tracklets)
            camera.add_tracker_callback(self.render.data.set_depth_tracklets)
            camera.start()

        self.correlation_streamer.add_stream_callback(self.render.data.set_correlation_stream)
        self.correlation_streamer.start()

        self.dtw_correlator.add_correlation_callback(self.correlation_streamer.add_correlation)
        self.dtw_correlator.start()

        self.pose_streamer.add_stream_callback(self.dtw_correlator.set_pose_stream)
        self.pose_streamer.add_stream_callback(self.render.data.set_pose_stream)
        self.pose_streamer.start()

        self.pose_detection.add_pose_callback(self.pose_streamer.add_pose)
        self.pose_detection.add_pose_callback(self.render.data.set_pose)
        self.pose_detection.start()

        self.tracker.add_tracklet_callback(self.pose_detection.add_tracklet)
        self.tracker.add_tracklet_callback(self.render.data.set_tracklet)
        self.tracker.start()

        if self.av:
            self.av.add_output_callback(self.render.data.set_light_image)
            self.av.start()

        # GUIGUIGUIGUIGUIGUIGUIGUIGUIGUIGUIGUI
        self.gui.exit_callback = self.stop

        for i in range(ceil(len(self.cameras) / 2.0)):
            c: int = i * 2
            if c + 1 < len(self.cameras):
                self.gui.addFrame([self.cameras[c].gui.get_gui_frame(), self.cameras[c+1].gui.get_gui_frame()])
            else:
                self.gui.addFrame([self.cameras[c].gui.get_gui_frame()])

        if self.av:
            self.gui.addFrame([self.av.gui.get_gui_frame()])

        if self.player:
            self.gui.addFrame([self.player.get_gui_frame(), self.tracker.gui.get_gui_frame()])
        if self.recorder:
            self.gui.addFrame([self.recorder.get_gui_frame(), self.tracker.gui.get_gui_frame()])
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

        self.is_running = True

        self.render.window_manager.add_exit_callback(self.stop)
        self.render.window_manager.add_keyboard_callback(self.render_keyboard_callback)
        self.render.window_manager.start()

    def stop(self) -> None:
        # print("Stopping main application...")
        if not self.is_running:
            return
        self.is_running = False

        # print("Stopping render...")
        self.render.window_manager.stop()

        # print("Stopping player...")
        if self.player:
            self.player.stop()

        # print('stop cameras')
        for camera in self.cameras:
            camera.stop()

        # print('stop tracker')
        self.tracker.stop()

        if self.pose_detection:
            self.pose_detection.stop()
        if self.pose_streamer:
            self.pose_streamer.stop()
        if self.dtw_correlator:
            self.dtw_correlator.stop()
        if self.correlation_streamer:
            self.correlation_streamer.stop()

        # print('stop av')
        if self.av:
            self.av.stop()

        # print('stop recorder')
        if self.recorder:
            self.recorder.stop()

        # print('stop gui')
        self.gui.stop()

        # print('join cameras')
        for camera in self.cameras:
            camera.join(timeout=8)

        self.is_finished = True
        # print("Main application stopped.")


    def render_keyboard_callback(self, key, x, y) -> None:
        if not  self.is_running: return
        if key == b'g' or key == b'G':
            if not self.gui or not self.gui.running: return
            self.gui.bringToFront()
