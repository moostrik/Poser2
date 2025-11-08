# Standard library imports
from math import ceil
from typing import Optional

# Local application imports
from modules.Settings import Settings
from modules.gui.PyReallySimpleGui import Gui

from modules.cam.DepthCam import DepthCam, DepthSimulator
from modules.cam.recorder.SyncRecorderGui import SyncRecorderGui as Recorder
from modules.cam.depthplayer.SyncPlayerGui import SyncPlayerGui as Player
from modules.cam.FrameSyncBang import FrameSyncBang

from modules.tracker.TrackerBase import TrackerType
from modules.tracker.panoramic.PanoramicTracker import PanoramicTracker
from modules.tracker.onepercam.OnePerCamTracker import OnePerCamTracker

from modules.pose.detection.PoseDetectionPipeline import PoseDetectionPipeline

from modules.pose import filters
from modules.pose.filters.PoseFilterPipelineTracker import PoseFilterPipelineTracker
from modules.pose.filters.general.gui.PoseSmootherGui import PoseSmootherGui

from modules.pose.correlation.PoseSimilarityComputer import PoseSimilarityComputer

from modules.pose.PoseStream import PoseStreamManager
from modules.pose.correlation.PoseStreamCorrelator import PoseStreamCorrelator

from modules.CaptureDataHub import CaptureDataHub
from modules.RenderDataHub import RenderDataHub
from modules.RenderDataHub_Old import RenderDataHub_Old

from modules.render.HDTRenderManager import HDTRenderManager
from modules.WS.WSPipeline import WSPipeline


class Main():
    def __init__(self, settings: Settings) -> None:

        self.settings: Settings = settings
        self.gui = Gui(settings)

        self.is_running: bool = False
        self.is_finished: bool = False

        # CAMERA
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
        self.frame_sync_bang = FrameSyncBang(settings, False, 'frame_sync')

        # TRACKER
        if settings.tracker_type == TrackerType.PANORAMIC:
            self.tracker = PanoramicTracker(self.gui, settings)
        else:
            self.tracker = OnePerCamTracker(self.gui, settings)
        self.tracklet_sync_bang = FrameSyncBang(settings, False, 'tracklet_sync')

        # POSE
        self.pose_detection = PoseDetectionPipeline(settings)

        self.pose_raw_filters = PoseFilterPipelineTracker(
            settings.num_players,
            [
                lambda: filters.PoseConfidenceFilter(filters.PoseConfidenceFilterConfig(settings.pose_conf_threshold)),
                filters.PoseAngleExtractor,
                filters.PoseDeltaExtractor
            ]
        )


        self.point_smooth_config = filters.PoseSmootherConfig()
        self.point_smooth_gui: PoseSmootherGui = PoseSmootherGui(self.point_smooth_config, self.gui, 'Point Smoother')

        self.pose_smooth_filters = PoseFilterPipelineTracker(
            settings.num_players,
            [
                lambda: filters.OldPosePointSmoother(self.point_smooth_config),
                lambda: filters.OldPoseAngleSmoother(self.point_smooth_config),
                lambda: filters.PoseBBoxSmoother(self.point_smooth_config),
                filters.PoseDeltaExtractor,
                filters.PoseMotionTimeAccumulator,
                lambda: filters.OldPoseDeltaSmoother(self.point_smooth_config)
            ]
        )

        self.pose_correlator: PoseSimilarityComputer = PoseSimilarityComputer(settings)

        self.pose_streamer = PoseStreamManager(settings)
        self.stream_correlator: Optional[PoseStreamCorrelator] = PoseStreamCorrelator(settings)

        # DATA
        self.capture_data_hub = CaptureDataHub()
        self.render_data_hub = RenderDataHub(settings)
        self.render_data_hub_old = RenderDataHub_Old(settings)

        # RENDER
        self.WS: Optional[WSPipeline] = None
        if settings.art_type == Settings.ArtType.WS:
            self.WS = WSPipeline(self.gui, settings)
            # self.render = WSRenderManager(self.gui, self.capture_data_hub, self.render_data_hub, settings)
        if settings.art_type == Settings.ArtType.HDT:
            self.render = HDTRenderManager(self.gui, self.capture_data_hub, self.render_data_hub, self.render_data_hub_old, settings)

    def start(self) -> None:

        for camera in self.cameras:

            camera.add_preview_callback(self.capture_data_hub.set_cam_image)
            if self.recorder:
                camera.add_sync_callback(self.recorder.set_synced_frames)
            camera.add_frame_callback(self.pose_detection.set_image)
            camera.add_frame_callback(self.frame_sync_bang.add_frame)
            camera.add_tracker_callback(self.tracker.add_cam_tracklets)
            camera.add_tracker_callback(self.capture_data_hub.set_cam_tracklets)
            camera.add_tracker_callback(self.tracklet_sync_bang.add_frame)
            camera.start()

        if self.stream_correlator:
            self.stream_correlator.add_correlation_callback(self.capture_data_hub.set_motion_correlation)
            self.stream_correlator.add_correlation_callback(self.render_data_hub_old.add_motion_correlation)
            self.stream_correlator.start()
            self.pose_streamer.add_stream_callback(self.stream_correlator.set_pose_stream)

        self.pose_correlator.add_correlation_callback(self.capture_data_hub.set_pose_correlation)
        self.pose_correlator.add_correlation_callback(self.render_data_hub_old.add_pose_correlation)
        self.pose_correlator.start()

        self.pose_streamer.add_stream_callback(     self.capture_data_hub.set_pose_stream)
        self.pose_streamer.start()

        self.pose_detection.add_callback(self.pose_raw_filters.add_poses)
        self.pose_raw_filters.add_callback(self.capture_data_hub.set_raw_poses)
        self.pose_raw_filters.add_callback(self.pose_smooth_filters.add_poses)
        self.pose_smooth_filters.add_callback(self.capture_data_hub.set_smooth_poses)
        self.pose_smooth_filters.add_callback(self.render_data_hub.add_poses)

        # DETECTION
        self.pose_detection.add_callback(self.render_data_hub_old.add_poses)
        self.pose_detection.start()

        self.tracker.add_tracklet_callback(self.pose_detection.set_tracklets)
        self.tracker.add_tracklet_callback(self.capture_data_hub.set_tracklets)
        self.tracker.start()

        self.tracklet_sync_bang.add_callback(self.tracker.notify_update)
        self.frame_sync_bang.add_callback(self.pose_detection.notify_update)

        if self.WS:
            self.pose_detection.add_callback(self.WS.add_poses)
            self.pose_streamer.add_stream_callback(self.WS.add_pose_stream)
            self.WS.add_output_callback(self.capture_data_hub.set_light_image)
            self.WS.start()

        # GUIGUIGUIGUIGUIGUIGUIGUIGUIGUIGUIGUI
        self.gui.exit_callback = self.stop

        for i in range(ceil(len(self.cameras) / 2.0)):
            c: int = i * 2
            if c + 1 < len(self.cameras):
                self.gui.addFrame([self.cameras[c].gui.get_gui_frame(), self.cameras[c+1].gui.get_gui_frame()])
            else:
                self.gui.addFrame([self.cameras[c].gui.get_gui_frame()])

        if self.WS:
            self.gui.addFrame([self.WS.gui.get_gui_frame(), self.WS.gui.get_gui_test_frame()])

        self.gui.addFrame([self.point_smooth_gui.get_gui_frame()])

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
        if self.stream_correlator:
            self.stream_correlator.stop()

        # print('stop av')
        if self.WS:
            self.WS.stop()

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
