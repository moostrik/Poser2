# Standard library imports
from math import ceil
from typing import Optional
from functools import partial

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


from modules.pose.detection import PointBatchExtractor, MMDetection
from modules.pose import nodes
from modules.pose import trackers
from modules.pose import gui

from modules.pose.similarity.SimilarityComputer import SimilarityComputer

from modules.pose.pd_stream.PDStream import PDStreamManager
from modules.pose.pd_stream.PDSimilarityComputer import PDStreamComputer

from modules.DataHub import DataHub, DataType
from modules.inout.SoundOSC import SoundOSC

from modules.render.HDTRenderManager import HDTRenderManager


class Main():
    def __init__(self, settings: Settings) -> None:

        self.settings: Settings = settings
        self.gui = Gui(settings.gui)
        num_players: int = settings.num_players

        self.is_running: bool = False
        self.is_finished: bool = False

        # CAMERA
        self.cameras: list[DepthCam | DepthSimulator] = []
        self.recorder: Optional[Recorder] = None
        self.player: Optional[Player] = None
        if settings.camera.sim_enabled:
            self.player = Player(self.gui, settings.camera)
            for cam_id in settings.camera.ids:
                self.cameras.append(DepthSimulator(self.gui, self.player, cam_id, settings.camera))
        else:
            self.recorder = Recorder(self.gui, settings.camera)
            for cam_id in settings.camera.ids:
                camera = DepthCam(self.gui, cam_id, settings.camera)
                self.cameras.append(camera)
        self.frame_sync_bang = FrameSyncBang(settings.camera, False, 'frame_sync')

        # TRACKER
        if settings.tracker_type == TrackerType.PANORAMIC:
            self.tracker = PanoramicTracker(self.gui, settings.num_players, settings.camera.num)
        else:
            self.tracker = OnePerCamTracker(self.gui, settings.num_players)
        self.tracklet_sync_bang = FrameSyncBang(settings.camera, False, 'tracklet_sync')

        # TRACKER POSE BRIDGE
        # a bridge that gathers tracklets and images and "bangs" the pose processing pipeline
        # similar to or in concert with FrameSyncBang and TrackletSyncBang

        # POSE DETECTOR
        self.pose_detector = MMDetection(settings.pose)

        # POSE CONFIGURATION
        self.image_crop_config =    nodes.ImageCropProcessorConfig(expansion=settings.pose.crop_expansion)
        self.prediction_config =    nodes.PredictorConfig(frequency=settings.camera.fps)

        self.b_box_smooth_config =  nodes.EuroSmootherConfig()
        self.point_smooth_config =  nodes.EuroSmootherConfig()
        self.angle_smooth_config =  nodes.EuroSmootherConfig()
        self.a_vel_smooth_config =  nodes.EuroSmootherConfig()

        self.b_box_interp_config =  nodes.LerpInterpolatorConfig(input_frequency=settings.camera.fps)
        self.point_interp_config =  nodes.ChaseInterpolatorConfig(input_frequency=settings.camera.fps)
        self.angle_interp_config =  nodes.ChaseInterpolatorConfig(input_frequency=settings.camera.fps)

        self.b_box_smooth_gui =     gui.EuroSmootherGui(self.b_box_smooth_config, self.gui, 'BBox')
        self.point_smooth_gui =     gui.EuroSmootherGui(self.point_smooth_config, self.gui, 'Point')
        self.angle_smooth_gui =     gui.EuroSmootherGui(self.angle_smooth_config, self.gui, 'Angle')
        self.a_vel_smooth_gui =     gui.EuroSmootherGui(self.a_vel_smooth_config, self.gui, 'Vel')

        # self.b_box_interp_gui =     gui.InterpolatorGui(self.b_box_interp_config, self.gui, 'BBox')
        self.point_interp_gui =     gui.InterpolatorGui(self.point_interp_config, self.gui, 'Point')
        self.angle_interp_gui =     gui.InterpolatorGui(self.angle_interp_config, self.gui, 'Angle')

        # POSE PROCESSING PIPELINES
        self.pose_from_tracklet =   trackers.PoseFromTrackletGenerator(num_players)

        self.bbox_filters =      trackers.FilterTracker(
            settings.num_players,
            [
                lambda: nodes.BBoxEuroSmoother(self.b_box_smooth_config),
                lambda: nodes.BBoxPredictor(self.prediction_config),
            ]
        )

        self.image_crop_processor = trackers.ImageCropProcessorTracker(num_players, self.image_crop_config)
        self.point_extractor =      PointBatchExtractor(self.pose_detector)

        self.pose_raw_filters =     trackers.FilterTracker(
            settings.num_players,
            [
                lambda: nodes.PointConfidenceFilter(nodes.ConfidenceFilterConfig(settings.pose.confidence_threshold)),
                nodes.AngleExtractor,
                nodes.AngleVelExtractor,
                lambda: nodes.PoseValidator(nodes.ValidatorConfig(name="Raw")),
            ]
        )

        self.pose_smooth_filters = trackers.FilterTracker(
            settings.num_players,
            [
                lambda: nodes.PointEuroSmoother(self.point_smooth_config),
                lambda: nodes.AngleEuroSmoother(self.angle_smooth_config),
                nodes.AngleVelExtractor,
                nodes.AngleSymExtractor,
                nodes.MotionTimeExtractor,
                nodes.AgeExtractor,
                lambda: nodes.PoseValidator(nodes.ValidatorConfig(name="Smooth")),
            ]
        )

        self.pose_prediction_filters = trackers.FilterTracker(
            settings.num_players,
            [
                lambda: nodes.PointPredictor(self.prediction_config),
                lambda: nodes.AnglePredictor(self.prediction_config),
                lambda: nodes.AngleStickyFiller(nodes.StickyFillerConfig(False, True)),
                lambda: nodes.PoseValidator(nodes.ValidatorConfig(name="Prediction")),
            ]
        )

        self.interpolator = trackers.InterpolatorTracker(
            settings.num_players,
            [
                lambda: nodes.BBoxLerpInterpolator(self.b_box_interp_config),
                lambda: nodes.PointChaseInterpolator(self.point_interp_config),
                lambda: nodes.AngleChaseInterpolator(self.angle_interp_config),
            ]
        )

        self.pose_interpolation_pipeline = trackers.FilterTracker(
            settings.num_players,
            [
                nodes.AngleVelExtractor,
                nodes.AngleSymExtractor,
                nodes.MotionTimeExtractor,
                nodes.AgeExtractor,
                lambda: nodes.AngleVelStickyFiller(nodes.StickyFillerConfig(False, False)),
                lambda: nodes.AngleVelEuroSmoother(self.a_vel_smooth_config),
                lambda: nodes.PoseValidator(nodes.ValidatorConfig(name="Interpolation")),
            ]
        )

        self.debug_tracker = trackers.DebugTracker(num_players)

        self.pose_similator: SimilarityComputer = SimilarityComputer()

        self.pd_pose_streamer = PDStreamManager(settings)
        self.pd_stream_similator: Optional[PDStreamComputer] = None


        # DATA
        self.data_hub = DataHub()
        self.sound_osc = SoundOSC(self.data_hub, settings)

        # RENDER
        # self.WS: Optional[WSPipeline] = None
        # if settings.art_type == Settings.ArtType.WS:
        #     self.WS = WSPipeline(self.gui, settings)
            # self.render = WSRenderManager(self.gui, self.capture_data_hub, self.render_data_hub, settings)
        if settings.art_type == Settings.ArtType.HDT:
            self.render = HDTRenderManager(self.gui, self.data_hub, settings)

    def start(self) -> None:

        for camera in self.cameras:

            camera.add_preview_callback(self.data_hub.set_cam_image)
            if self.recorder:
                camera.add_sync_callback(self.recorder.set_synced_frames)
            camera.add_frame_callback(self.image_crop_processor.set_image)
            camera.add_frame_callback(self.frame_sync_bang.add_frame)
            camera.add_tracker_callback(self.tracker.add_cam_tracklets)
            camera.add_tracker_callback(self.data_hub.set_depth_tracklets)
            camera.add_tracker_callback(self.tracklet_sync_bang.add_frame)
            camera.start()

        if self.pd_stream_similator:
            self.pd_pose_streamer.add_stream_callback(self.pd_stream_similator.set_pose_stream)
            self.pd_stream_similator.add_correlation_callback(self.data_hub.set_motion_similarity)
            self.pd_stream_similator.start()

        self.pd_pose_streamer.add_stream_callback(self.data_hub.set_pd_stream)
        self.pd_pose_streamer.start()

        self.pose_similator.add_correlation_callback(self.data_hub.set_pose_similarity)
        self.pose_similator.start()

        # POSE PROCESSING PIPELINES
        self.pose_from_tracklet.add_poses_callback(self.bbox_filters.process)
        self.bbox_filters.add_poses_callback(self.image_crop_processor.process)
        self.image_crop_processor.add_image_callback(self.point_extractor.set_images)
        self.image_crop_processor.add_poses_callback(self.point_extractor.process)
        self.point_extractor.add_poses_callback(self.pose_raw_filters.process)
        self.pose_raw_filters.add_poses_callback(self.pd_pose_streamer.submit)
        self.pose_raw_filters.add_poses_callback(partial(self.data_hub.set_poses, DataType.pose_R)) # raw poses

        self.pose_raw_filters.add_poses_callback(self.pose_smooth_filters.process)
        self.pose_smooth_filters.add_poses_callback(self.pose_similator.submit)
        self.pose_smooth_filters.add_poses_callback(self.pose_prediction_filters.process)
        self.pose_prediction_filters.add_poses_callback(partial(self.data_hub.set_poses, DataType.pose_S)) # smooth poses

        self.pose_prediction_filters.add_poses_callback(self.interpolator.submit)
        self.interpolator.add_poses_callback(self.pose_interpolation_pipeline.process)
        self.pose_interpolation_pipeline.add_poses_callback(partial(self.data_hub.set_poses, DataType.pose_I)) # interpolated poses

        # self.pose_interpolation_pipeline.add_poses_callback(self.debug_tracker.process)

        self.data_hub.add_update_callback(self.interpolator.update)


        self.sound_osc.start()
        self.data_hub.add_update_callback(self.sound_osc.notify_update)

        # DETECTION
        self.pose_detector.start()

        # TRACKER
        self.tracker.add_tracklet_callback(self.pose_from_tracklet.submit_tracklets)
        self.tracker.add_tracklet_callback(self.data_hub.set_tracklets)
        self.tracker.start()

        self.tracklet_sync_bang.add_callback(self.tracker.notify_update)
        self.frame_sync_bang.add_callback(self.pose_from_tracklet.update)

        # if self.WS:
        #     self.pose_detection.add_callback(self.WS.add_poses)
        #     self.pose_streamer.add_stream_callback(self.WS.add_pose_stream)
        #     self.WS.add_output_callback(self.capture_data_hub.set_light_image)
        #     self.WS.start()

        # GUIGUIGUIGUIGUIGUIGUIGUIGUIGUIGUIGUI
        self.gui.exit_callback = self.stop

        for i in range(ceil(len(self.cameras) / 2.0)):
            c: int = i * 2
            if c + 1 < len(self.cameras):
                self.gui.addFrame([self.cameras[c].gui.get_gui_frame(), self.cameras[c+1].gui.get_gui_frame()])
            else:
                self.gui.addFrame([self.cameras[c].gui.get_gui_frame()])

        # if self.WS:
        #     self.gui.addFrame([self.WS.gui.get_gui_frame(), self.WS.gui.get_gui_test_frame()])

        self.gui.addFrame([self.b_box_smooth_gui.get_gui_frame()])
        self.gui.addFrame([self.point_smooth_gui.get_gui_frame(), self.point_interp_gui.get_gui_frame()])
        self.gui.addFrame([self.angle_smooth_gui.get_gui_frame(), self.angle_interp_gui.get_gui_frame()])
        self.gui.addFrame([self.a_vel_smooth_gui.get_gui_frame()])

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

        if self.pose_detector:
            self.pose_detector.stop()
        if self.pd_pose_streamer:
            self.pd_pose_streamer.stop()
        if self.pd_stream_similator:
            self.pd_stream_similator.stop()

        # print('stop av')
        # if self.WS:
        #     self.WS.stop()

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
