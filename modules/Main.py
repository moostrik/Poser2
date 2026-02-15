# Standard library imports
from math import ceil
from typing import Optional
from functools import partial
from datetime import datetime
import os

# Local application imports
from modules.render.RenderManager import RenderManager
from modules.Settings import Settings
from modules.DataHub import DataHub, Stage
from modules.pose.Frame import FrameField
from modules.gui import Gui
from modules.gui.ConfigGuiGenerator import ConfigGuiGenerator
from modules.inout import OscSound, ArtNetLed
from modules.cam import DepthCam, DepthSimulator, Recorder, Player, FrameSyncBang
from modules.tracker import TrackerType, PanoramicTracker, OnePerCamTracker
from modules.pose import batch, guis, nodes, trackers
from modules.utils import Timer, TimerConfig

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

        # DATA
        self.data_hub = DataHub()
        self.sound_osc = OscSound(self.data_hub, settings.sound_osc)

        # ARTNET LED CONTROLLERS
        self.artnet_configs = settings.artnet_leds
        self.artnet_controllers = [ArtNetLed(cfg) for cfg in self.artnet_configs]
        self.artnet_guis = [
            ConfigGuiGenerator(self.artnet_configs[0], self.gui, "ArtNet Bar 1"),
            ConfigGuiGenerator(self.artnet_configs[1], self.gui, "ArtNet Bar 2"),
            ConfigGuiGenerator(self.artnet_configs[2], self.gui, "ArtNet Bar 3"),
        ]

        # TIMER
        self.timer_config = TimerConfig(duration=30.0, intermezzo=5.0)
        self.timer = Timer(self.timer_config)
        self.timer_gui = ConfigGuiGenerator(self.timer_config, self.gui, "Timer")

        # RENDER
        self.render = RenderManager(self.gui, self.data_hub, settings.render)
        self.data_gui = ConfigGuiGenerator(settings.render, self.gui, "Render", 4)

        # POSE CONFIGURATION
        # self.gpu_crop_config =      batch.GPUCropProcessorConfig(expansion_width=settings.pose.crop_expansion_width, expansion_height=settings.pose.crop_expansion_height, output_width=384, output_height=512, max_poses=settings.pose.max_poses)
        self.gpu_crop_config =      batch.ImageCropConfig(expansion_width=settings.pose.crop_expansion_width, expansion_height=settings.pose.crop_expansion_height, output_width=768, output_height=1024, max_poses=settings.pose.max_poses, verbose=False, enable_prev_crop=False)
        self.prediction_config =    nodes.PredictorConfig(frequency=settings.camera.fps)

        self.b_box_smooth_config =  nodes.EuroSmootherConfig()
        self.b_box_rate_config =    nodes.RateLimiterConfig(max_increase= 10, max_decrease= 0.2)
        self.point_smooth_config =  nodes.EuroSmootherConfig()
        self.angle_smooth_config =  nodes.EuroSmootherConfig()
        self.a_vel_smooth_config =  nodes.EuroSmootherConfig()
        self.simil_smooth_config =  nodes.EuroSmootherConfig()

        self.b_box_smooth_gui =     guis.EuroSmootherGui(self.b_box_smooth_config, self.gui, 'BBOX')
        self.point_smooth_gui =     guis.EuroSmootherGui(self.point_smooth_config, self.gui, 'POINT')
        self.angle_smooth_gui =     guis.EuroSmootherGui(self.angle_smooth_config, self.gui, 'ANGLE')
        self.a_vel_smooth_gui =     guis.EuroSmootherGui(self.a_vel_smooth_config, self.gui, 'ANGLE VEL')
        self.simil_smooth_gui =     guis.EuroSmootherGui(self.simil_smooth_config, self.gui, 'SIMILARITY')

        # self.b_box_interp_config =  nodes.LerpInterpolatorConfig(input_frequency=settings.camera.fps)
        self.b_box_interp_config =  nodes.ChaseInterpolatorConfig(input_frequency=settings.camera.fps)
        self.point_interp_config =  nodes.ChaseInterpolatorConfig(input_frequency=settings.camera.fps)
        self.angle_interp_config =  nodes.ChaseInterpolatorConfig(input_frequency=settings.camera.fps)
        self.simil_interp_config =  nodes.ChaseInterpolatorConfig(input_frequency=settings.camera.fps)

        self.b_box_interp_gui =     guis.InterpolatorGui(self.b_box_interp_config, self.gui, 'BBOX')
        self.point_interp_gui =     guis.InterpolatorGui(self.point_interp_config, self.gui, 'POINT')
        self.angle_interp_gui =     guis.InterpolatorGui(self.angle_interp_config, self.gui, 'ANGLE')
        self.simil_interp_gui =     guis.InterpolatorGui(self.simil_interp_config, self.gui, 'SIMILARITY')

        # self.motion_smooth_config = nodes.EmaSmootherConfig(attack=0.95, release=0.8)
        # self.motion_smooth_gui =    guis.EmaSmootherGui(self.motion_smooth_config, self.gui, 'MOTION')
        self.motion_ma_config =     nodes.MovingAverageConfig(window_size=30, window_type=nodes.WindowType.TRIANGULAR)
        self.motion_ma_gui =        guis.MovingAverageSmootherGui(self.motion_ma_config, self.gui, 'MOTION_MA')
        self.motion_easing_config = nodes.EasingConfig(easing_name='easeInOutSine')
        self.motion_easing_gui =    guis.EasingGui(self.motion_easing_config, self.gui, 'MOTION_EASE')
        self.motion_extractor_config = nodes.AngleMotionExtractorConfig(noise_threshold=0.05, max_threshold=0.5)
        self.motion_extractor_gui = guis.AngleMotionExtractorGui(self.motion_extractor_config, self.gui, 'MOTION_EXT')

        # POSE PROCESSING PIPELINES
        self.poses_from_tracklets = batch.PosesFromTracklets(num_players)

        self.gpu_crop_processor =   batch.ImageCropProcessor(self.gpu_crop_config)
        self.point_extractor =      batch.PointBatchExtractor(settings.pose)  # GPU-based 2D point extractor
        self.mask_extractor =       batch.MaskBatchExtractor(settings.pose)   # GPU-based segmentation mask extractor
        self.flow_extractor =       batch.FlowBatchExtractor(settings.pose)   # GPU-based optical flow extractor

        self.window_similator_config = batch.WindowSimilarityConfig(window_length=int(0.5 * settings.camera.fps))
        self.window_similator=      batch.WindowSimilarity(self.window_similator_config)
        self.window_similarity_gui = guis.WindowSimilarityGui(self.window_similator_config, self.gui, 'SIMILARITY')

        # Feature applicators (replace SimilarityExtractor)
        self.similarity_applicator = nodes.SimilarityApplicator(max_poses=settings.pose.max_poses)
        self.leader_applicator =     nodes.LeaderScoreApplicator(max_poses=settings.pose.max_poses)
        self.motion_gate_applicator = nodes.MotionGateApplicator(nodes.MotionGateApplicatorConfig(max_poses=settings.pose.max_poses))
        self.motion_gate_tracker =   trackers.FilterTracker(num_players, [lambda: self.motion_gate_applicator])

        self.debug_tracker =        trackers.DebugTracker(num_players)

        # WINDOW TRACKERS
        self.window_tracker_R =     trackers.AllWindowTracker(num_players, trackers.WindowNodeConfig(window_size=int(6.0 * settings.camera.fps)))
        self.window_tracker_S =     trackers.AllWindowTracker(num_players, trackers.WindowNodeConfig(window_size=int(6.0 * settings.camera.fps)))
        self.window_tracker_I =     trackers.AllWindowTracker(num_players, trackers.WindowNodeConfig(window_size=int(6.0 * settings.render.fps)))

        self.bbox_filters =      trackers.FilterTracker(
            settings.num_players,
            [
                lambda: nodes.BBoxEuroSmoother(self.b_box_smooth_config),
                lambda: nodes.BBoxPredictor(self.prediction_config),
                # lambda: nodes.BBoxRateLimiter(self.b_box_rate_config),
            ]
        )

        self.pose_raw_filters =     trackers.FilterTracker(
            settings.num_players,
            [
                lambda: nodes.PointDualConfFilter(nodes.DualConfFilterConfig(settings.pose.confidence_low, settings.pose.confidence_high)),
                # lambda: nodes.PointTemporalStabilizer(nodes.TemporalStabilizerConfig()),
                nodes.AngleExtractor,
                lambda: nodes.AngleVelExtractor(fps=settings.camera.fps),
                # lambda: nodes.PoseValidator(nodes.ValidatorConfig(name="Raw")),
            ]
        )

        self.pose_smooth_filters = trackers.FilterTracker(
            settings.num_players,
            [
                lambda: nodes.PointEuroSmoother(self.point_smooth_config),
                nodes.AngleExtractor,
                lambda: nodes.AngleVelExtractor(fps=settings.camera.fps),
                lambda: nodes.AngleVelEuroSmoother(self.angle_smooth_config),
                lambda: nodes.AngleEuroSmoother(self.angle_smooth_config),
                # lambda: nodes.AngleStickyFiller(nodes.StickyFillerConfig(init_to_zero=False, hold_scores=False)),
                lambda: nodes.AngleMotionExtractor(self.motion_extractor_config),
                # lambda: nodes.AngleMotionEmaSmoother(self.motion_smooth_config),
                # lambda: nodes.AngleMotionEasingNode(self.motion_easing_config),
                nodes.AngleSymExtractor,
                nodes.MotionTimeExtractor,
                nodes.AgeExtractor,
                lambda: self.similarity_applicator,
                lambda: self.leader_applicator,
                lambda: nodes.SimilarityEuroSmoother(self.simil_smooth_config),
                # lambda: nodes.PoseValidator(nodes.ValidatorConfig(name="Smooth")),
            ]
        )


        self.pose_prediction_filters = trackers.FilterTracker(
            settings.num_players,
            [
                lambda: nodes.PointPredictor(self.prediction_config),
                lambda: nodes.AnglePredictor(self.prediction_config),
                lambda: nodes.AngleVelPredictor(self.prediction_config),
                lambda: nodes.AngleStickyFiller(nodes.StickyFillerConfig(init_to_zero=False, hold_scores=True)),
                lambda: nodes.SimilarityStickyFiller(nodes.StickyFillerConfig(init_to_zero=True, hold_scores=False)),
                # lambda: nodes.PoseValidator(nodes.ValidatorConfig(name="Prediction")),
            ]
        )

        self.interpolator = trackers.InterpolatorTracker(
            settings.num_players,
            [
                lambda: nodes.BBoxChaseInterpolator(self.b_box_interp_config),
                lambda: nodes.PointChaseInterpolator(self.point_interp_config),
                lambda: nodes.AngleChaseInterpolator(self.angle_interp_config),
                lambda: nodes.AngleVelChaseInterpolator(self.angle_interp_config),
                lambda: nodes.SimilarityChaseInterpolator(self.simil_interp_config),
            ]
        )

        self.pose_interpolation_pipeline = trackers.FilterTracker(
            settings.num_players,
            [
                # lambda: nodes.AngleVelExtractor(fps=settings.render.fps),
                nodes.AngleSymExtractor,
                nodes.MotionTimeExtractor,
                nodes.AgeExtractor,
                lambda: nodes.AngleVelStickyFiller(nodes.StickyFillerConfig(init_to_zero=True, hold_scores=False)),
                lambda: nodes.AngleVelEuroSmoother(self.a_vel_smooth_config),
                lambda: nodes.AngleMotionExtractor(self.motion_extractor_config),
                lambda: nodes.AngleMotionMovingAverageSmoother(self.motion_ma_config),
                # lambda: nodes.PoseValidator(nodes.ValidatorConfig(name="Interpolation")),
            ]
        )

    def start(self) -> None:
        for camera in self.cameras:

            camera.add_preview_callback(self.data_hub.set_cam_image)
            if self.recorder:
                camera.add_sync_callback(self.recorder.set_synced_frames)
            camera.add_frame_callback(self.gpu_crop_processor.set_image)
            camera.add_frame_callback(self.frame_sync_bang.add_frame)
            camera.add_tracker_callback(self.tracker.add_cam_tracklets)
            camera.add_tracker_callback(self.data_hub.set_depth_tracklets)
            camera.add_tracker_callback(self.tracklet_sync_bang.add_frame)
            camera.start()

        # BBOX
        self.poses_from_tracklets.add_poses_callback(self.bbox_filters.process)
        self.bbox_filters.add_poses_callback(self.gpu_crop_processor.process)

        # POSE RAW
        self.point_extractor.add_poses_callback(self.pose_raw_filters.process)
        self.pose_raw_filters.add_poses_callback(partial(self.data_hub.set_poses, Stage.RAW))
        self.pose_raw_filters.add_poses_callback(self.window_tracker_R.process)
        self.window_tracker_R.add_callback(partial(self.data_hub.set_feature_windows, Stage.RAW))

        # POSE SMOOTH
        self.pose_raw_filters.add_poses_callback(self.pose_smooth_filters.process)

        # POSE PREDICTION
        self.pose_smooth_filters.add_poses_callback(self.pose_prediction_filters.process)
        self.pose_prediction_filters.add_poses_callback(partial(self.data_hub.set_poses, Stage.SMOOTH))
        self.pose_smooth_filters.add_poses_callback(self.window_tracker_S.process)
        self.window_tracker_S.add_callback(partial(self.data_hub.set_feature_windows, Stage.SMOOTH))

        # INTERPOLATION
        self.data_hub.add_update_callback(self.interpolator.update)
        self.pose_prediction_filters.add_poses_callback(self.interpolator.submit)
        self.interpolator.add_poses_callback(self.pose_interpolation_pipeline.process)

        # MOTION GATE (after interpolation pipeline, before DataHub)
        self.pose_interpolation_pipeline.add_poses_callback(self.motion_gate_applicator.submit) # dit slaat nergens op??
        self.pose_interpolation_pipeline.add_poses_callback(self.motion_gate_tracker.process)
        self.motion_gate_tracker.add_poses_callback(partial(self.data_hub.set_poses, Stage.LERP))
        self.motion_gate_tracker.add_poses_callback(self.window_tracker_I.process)
        self.window_tracker_I.add_callback(partial(self.data_hub.set_feature_windows, Stage.LERP))

        # SIMILARITY COMPUTATION (uses combined callback for motion gate and velocity weighting)
        self.window_tracker_S.add_callback(self.window_similator.submit_all)
        self.window_similator.add_callback(lambda result: self.similarity_applicator.submit(result[0]))
        self.window_similator.add_callback(lambda result: self.leader_applicator.submit(result[1]))
        self.window_similator.start()

        # POSE ESTIMATION
        self.gpu_crop_processor.add_callback(self.point_extractor.process)
        self.point_extractor.start()

        # SEGMENTATION
        self.gpu_crop_processor.add_callback(self.mask_extractor.process)
        self.mask_extractor.add_callback(self.data_hub.set_gpu_frames)
        self.mask_extractor.start()

        # FLOW
        self.gpu_crop_processor.add_callback(self.flow_extractor.process)
        self.flow_extractor.add_callback(self.data_hub.set_flow_tensors)
        self.flow_extractor.start()

        # TRACKER
        self.tracker.add_tracklet_callback(self.poses_from_tracklets.submit_tracklets)
        self.tracker.add_tracklet_callback(self.data_hub.set_tracklets)
        self.tracker.start()

        self.tracklet_sync_bang.add_callback(self.tracker.notify_update)
        self.frame_sync_bang.add_callback(self.poses_from_tracklets.generate)

        # IN / OUT
        self.sound_osc.start()
        self.data_hub.add_update_callback(self.sound_osc.notify_update)

        for artnet in self.artnet_controllers:
            artnet.start()

        # TIMER
        self.timer.start()
        self.timer.add_time_callback(lambda t: self.data_hub.set_timer_time(t))
        self.timer.add_state_callback(lambda s: self.data_hub.set_timer_state(s))

        # GUIGUIGUIGUIGUIGUIGUIGUIGUIGUIGUIGUI
        self.gui.exit_callback = self.stop

        for i in range(ceil(len(self.cameras) / 3.0)):
            c: int = i * 3
            if c + 2 < len(self.cameras):
                self.gui.addFrame([self.cameras[c].gui.get_gui_frame(), self.cameras[c+1].gui.get_gui_frame(), self.cameras[c+2].gui.get_gui_frame()])
            elif c + 1 < len(self.cameras):
                self.gui.addFrame([self.cameras[c].gui.get_gui_frame(), self.cameras[c+1].gui.get_gui_frame()])
            else:
                self.gui.addFrame([self.cameras[c].gui.get_gui_frame()])

        self.gui.addFrame([self.artnet_guis[0].frame, self.artnet_guis[1].frame, self.artnet_guis[2].frame])
        self.gui.addFrame([self.b_box_smooth_gui.get_gui_frame(), self.b_box_interp_gui.get_gui_frame(), self.data_gui.frame])
        self.gui.addFrame([self.point_smooth_gui.get_gui_frame(), self.point_interp_gui.get_gui_frame()])
        self.gui.addFrame([self.angle_smooth_gui.get_gui_frame(), self.angle_interp_gui.get_gui_frame(), self.a_vel_smooth_gui.get_gui_frame()])
        self.gui.addFrame([self.motion_extractor_gui.get_gui_frame(), self.motion_ma_gui.get_gui_frame()])
        self.gui.addFrame([self.window_similarity_gui.get_gui_frame(), self.simil_smooth_gui.get_gui_frame(), self.simil_interp_gui.get_gui_frame()])
        if self.player:
            self.gui.addFrame([self.player.get_gui_frame(), self.tracker.gui.get_gui_frame(), self.timer_gui.frame])
        if self.recorder:
            self.gui.addFrame([self.recorder.get_gui_frame(), self.tracker.gui.get_gui_frame(), self.timer_gui.frame])
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
        if not self.is_running:
            return
        self.is_running = False

        self.render.window_manager.stop()

        if self.player:
            self.player.stop()
        for camera in self.cameras:
            camera.stop()
        if self.recorder:
            self.recorder.stop()

        self.tracker.stop()
        self.sound_osc.stop()

        for artnet in self.artnet_controllers:
            artnet.stop()

        self.timer.stop()

        self.point_extractor.stop()
        self.mask_extractor.stop()
        self.flow_extractor.stop()

        self.window_similator.stop()


        self.gui.stop()

        for camera in self.cameras:
            camera.join(timeout=10)

        self.is_finished = True

    def render_keyboard_callback(self, key, x, y) -> None:
        if not  self.is_running: return
        if key == b'g' or key == b'G':
            if not self.gui or not self.gui.running: return
            self.gui.bringToFront()
