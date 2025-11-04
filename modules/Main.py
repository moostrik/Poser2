# Standard library imports
from math import ceil
from typing import Optional

# Local application imports
from modules.CaptureDataHub import CaptureDataHub
from modules.cam.DepthCam import DepthCam, DepthSimulator
from modules.cam.recorder.SyncRecorderGui import SyncRecorderGui as Recorder
from modules.cam.depthplayer.SyncPlayerGui import SyncPlayerGui as Player
from modules.cam.FrameSyncBang import FrameSyncBang
from modules.gui.PyReallySimpleGui import Gui
from modules.pose.correlation.PoseSimilarityComputer import PoseSimilarityComputer
from modules.pose.correlation.PoseStreamCorrelator import PoseStreamCorrelator
from modules.RenderDataHub import RenderDataHub
from modules.pose.detection.PoseDetectionPipeline import PosePipeline
from modules.pose.PoseStream import PoseStreamManager
import modules.pose.filters as pose_filters
from modules.render.HDTRenderManager import HDTRenderManager
from modules.Settings import Settings
from modules.tracker.TrackerBase import TrackerType
from modules.tracker.panoramic.PanoramicTracker import PanoramicTracker
from modules.tracker.onepercam.OnePerCamTracker import OnePerCamTracker
from modules.WS.WSPipeline import WSPipeline


class Main():
    def __init__(self, settings: Settings) -> None:
        self.gui = Gui(settings)

        self.settings: Settings = settings

        self.capture_data_hub = CaptureDataHub()
        self.render_data_hub = RenderDataHub(settings)

        self.WS: Optional[WSPipeline] = None
        # self.render = WSRenderManager(self.gui, self.capture_data_hub, self.render_data_hub, settings)
        if settings.art_type == Settings.ArtType.WS:
            self.WS = WSPipeline(self.gui, settings)
        if settings.art_type == Settings.ArtType.HDT:
            self.render = HDTRenderManager(self.gui, self.capture_data_hub, self.render_data_hub, settings)

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

        self.pose_confidence_filter =   pose_filters.PoseConfidenceFilter(settings)
        self.pose_angle_extractor =     pose_filters.PoseAngleExtractor()
        self.pose_delta_extractor_R =   pose_filters.PoseDeltaExtractor()
        self.pose_delta_extractor_S =   pose_filters.PoseDeltaExtractor()
        self.pose_mt_accumulator =      pose_filters.PoseMotionTimeAccumulator()

        self.pose_point_smoother =  pose_filters.PosePointSmoother(settings, 'POINT SMOOTHER', self.gui)
        self.pose_angle_smoother =  pose_filters.PoseAngleSmoother(settings, 'ANGLE SMOOTHER', self.gui)
        self.pose_bbox_smoother =   pose_filters.PoseBBoxSmoother(settings, 'BBOX SMOOTHER', self.gui)
        self.pose_delta_smoother =  pose_filters.PoseAngleDeltaSmoother(settings, 'DELTA SMOOTHER', self.gui)

        self.pose_raw_node =    pose_filters.PosePassThrough()
        self.pose_smooth_node = pose_filters.PosePassThrough()

        self.pose_correlator: PoseSimilarityComputer = PoseSimilarityComputer(settings)
        self.motion_correlator: Optional[PoseStreamCorrelator] = PoseStreamCorrelator(settings)

        self.frame_sync_bang = FrameSyncBang(settings, False, 'frame_sync')
        self.tracklet_sync_bang = FrameSyncBang(settings, False, 'tracklet_sync')

        self.is_running: bool = False
        self.is_finished: bool = False

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

        if self.motion_correlator:
            self.motion_correlator.add_correlation_callback(self.capture_data_hub.set_motion_correlation)
            self.motion_correlator.add_correlation_callback(self.render_data_hub.add_motion_correlation)
            self.motion_correlator.start()
            self.pose_streamer.add_stream_callback(self.motion_correlator.set_pose_stream)

        self.pose_correlator.add_correlation_callback(self.capture_data_hub.set_pose_correlation)
        self.pose_correlator.add_correlation_callback(self.render_data_hub.add_pose_correlation)
        self.pose_correlator.start()

        self.pose_streamer.add_stream_callback(     self.capture_data_hub.set_pose_stream)
        self.pose_streamer.start()

        # FILTERS RAW
        # filter by confidence
        self.pose_detection.add_callback(           self.pose_confidence_filter.add_poses)
        # extract angles
        self.pose_confidence_filter.add_callback(   self.pose_angle_extractor.add_poses)
        # extract raw angle deltas
        self.pose_angle_extractor.add_callback(     self.pose_delta_extractor_R.add_poses)
        # send 'raw' poses to the raw node
        self.pose_delta_extractor_R.add_callback(   self.pose_raw_node.add_poses)

        # FILTERS SMOOTHED
        # smooth points
        self.pose_angle_extractor.add_callback(     self.pose_point_smoother.add_poses) # don't use raw deltas
        # smooth angles
        self.pose_point_smoother.add_callback(      self.pose_angle_smoother.add_poses)
        # smooth bbox
        self.pose_angle_smoother.add_callback(      self.pose_bbox_smoother.add_poses)
        # extract smoothed angle deltas
        self.pose_bbox_smoother.add_callback(       self.pose_delta_extractor_S.add_poses)
        # extract movement time
        self.pose_delta_extractor_S.add_callback(   self.pose_mt_accumulator.add_poses)
        # smooth angle deltas
        self.pose_mt_accumulator.add_callback(      self.pose_delta_smoother.add_poses)
        # send smoothed poses to the smooth node
        self.pose_delta_smoother.add_callback(      self.pose_smooth_node.add_poses) # SMOOTHED POSES

        # CONNECT NODES TO DATA HUBS AND STREAMERS
        # RAW POSES
        self.pose_raw_node.add_callback(    self.capture_data_hub.set_raw_poses)
        self.pose_raw_node.add_callback(    self.pose_streamer.add_poses)
        self.pose_raw_node.add_callback(    self.pose_correlator.add_poses)
        # SMOOTHED POSES
        self.pose_smooth_node.add_callback( self.capture_data_hub.set_smooth_poses) # SMOOTHED POSES


        # DETECTION
        self.pose_detection.add_callback(self.render_data_hub.add_poses)
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

        self.gui.addFrame([self.pose_point_smoother.get_gui_frame(), self.pose_bbox_smoother.get_gui_frame()])
        self.gui.addFrame([self.pose_angle_smoother.get_gui_frame(), self.pose_delta_smoother.get_gui_frame()])

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
        if self.motion_correlator:
            self.motion_correlator.stop()

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
