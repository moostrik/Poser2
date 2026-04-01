# Standard library imports
from typing import Optional
from functools import partial

# Local application imports
from modules.main_settings import MainSettings
from modules.oak import Camera, FrameSync, Simulator, Player, Recorder
from modules.settings import presets, NiceServer
from modules.render import RenderManager
from modules.data_hub import DataHub, Stage
from modules.inout import OscSound, ArtNetBars, ArtNetBarsSettings
from modules.tracker import OnePerCamTracker
from modules.pose import batch, nodes, trackers
from modules.utils import Timer


class Main():
    def __init__(self, simulation: bool = False) -> None:

        self.new_settings = MainSettings()

        if not presets.load(self.new_settings, presets.startup_path()):
            presets.save(self.new_settings, presets.startup_path())  # create default preset if loading failed

        # CLI override: --simulation flag takes precedence over preset
        self.new_settings.camera.sim_enabled = simulation

        self.new_settings.initialize()

        num_players: int = self.new_settings.num_players
        cam_settings = self.new_settings.camera

        # Create controllers after preset loading so INIT fields (num_pixels, ip, etc.) are final
        self.artnet_controllers = [ArtNetBars(cfg) for cfg in self.new_settings.inout.children.values() if isinstance(cfg, ArtNetBarsSettings)]

        self.is_running: bool = False
        self.is_finished: bool = False

        # CAMERA
        self.cameras: list[Camera | Simulator] = []
        self.recorder: Optional[Recorder] = None
        self.player: Optional[Player] = None
        if cam_settings.sim_enabled:
            self.player = Player(cam_settings.simulator)
            for i in range(cam_settings.num_cameras):
                self.cameras.append(Simulator(self.player, cam_settings.cameras[i], cam_settings.simulator))
        else:
            self.recorder = Recorder(cam_settings.recorder)
            for i in range(cam_settings.num_cameras):
                camera = Camera(cam_settings.cameras[i])
                self.cameras.append(camera)
        self.frame_sync_bang = FrameSync(cam_settings, False, 'frame_sync')

        # TRACKER
        self.tracker = OnePerCamTracker(self.new_settings.tt.tracker, num_players)
        self.tracklet_sync_bang = FrameSync(cam_settings, False, 'tracklet_sync')

        # DATA
        self.data_hub = DataHub()
        self.sound_osc = OscSound(self.data_hub, self.new_settings.inout.osc_sound)

        # TIMER
        self.timer = Timer(self.new_settings.tt.timer)

        self.render_settings = self.new_settings.render
        self.render = RenderManager(self.data_hub, self.render_settings, num_cams=len(self.cameras), num_players=num_players)


        # Start settings server
        self.settings_server = NiceServer(self.new_settings, self.new_settings.server, on_exit=self.stop)
        self.settings_server.start()

        # POSE CONFIGURATION
        pose_settings = self.new_settings.pose.pose
        pose_group = self.new_settings.pose
        # self.gpu_crop_config =      batch.GPUCropProcessorConfig(expansion_width=pose_settings.crop_expansion_width, expansion_height=pose_settings.crop_expansion_height, output_width=384, output_height=512, max_poses=pose_settings.max_poses)
        self.gpu_crop_config =      batch.ImageCropConfig(expansion_width=pose_settings.crop_expansion_width, expansion_height=pose_settings.crop_expansion_height, output_width=768, output_height=1024, max_poses=pose_settings.max_poses, verbose=False, enable_prev_crop=False)
        self.b_box_smooth_config =  pose_group.bbox.smoother
        self.b_box_rate_config =    nodes.RateLimiterConfig(max_increase= 10, max_decrease= 0.2)
        self.point_smooth_config =  pose_group.point.smoother
        self.angle_smooth_config =  pose_group.angle.smoother
        self.a_vel_smooth_config =  pose_group.angle.angle_vel_smoother
        self.simil_smooth_config =  pose_group.similarity.smoother

        self.b_box_interp_config =  pose_group.bbox.interpolator
        self.point_interp_config =  pose_group.point.interpolator
        self.angle_interp_config =  pose_group.angle.interpolator
        self.simil_interp_config =  pose_group.similarity.interpolator

        self.motion_ma_config =     pose_group.motion.moving_average
        self.motion_easing_config = nodes.EasingConfig(easing_name='easeInOutSine')
        self.motion_extractor_config = pose_group.motion.extractor

        # POSE PROCESSING PIPELINES
        self.poses_from_tracklets = batch.PosesFromTracklets(num_players)

        self.gpu_crop_processor =   batch.ImageCropProcessor(self.gpu_crop_config)
        self.point_extractor =      batch.PointBatchExtractor(pose_settings)  # GPU-based 2D point extractor
        self.mask_extractor =       batch.MaskBatchExtractor(pose_settings)   # GPU-based segmentation mask extractor
        self.flow_extractor =       batch.FlowBatchExtractor(pose_settings)   # GPU-based optical flow extractor

        self.window_similator_config = self.new_settings.pose.window_similarity
        self.window_similator=      batch.WindowSimilarity(self.window_similator_config)

        self.window_correlator_config = self.new_settings.pose.window_correlation
        self.window_correlator =    batch.WindowCorrelation(self.window_correlator_config)

        # Feature applicators (replace SimilarityExtractor)
        self.similarity_applicator = nodes.SimilarityApplicator(max_poses=pose_settings.max_poses)
        self.leader_applicator =     nodes.LeaderScoreApplicator(max_poses=pose_settings.max_poses)
        self.motion_gate_applicator = nodes.MotionGateApplicator(nodes.MotionGateApplicatorConfig(max_poses=pose_settings.max_poses))
        self.motion_gate_tracker =   trackers.FilterTracker(num_players, [lambda: self.motion_gate_applicator])

        self.debug_tracker =        trackers.DebugTracker(num_players)

        # WINDOW TRACKERS
        self.window_tracker_R =     trackers.AllWindowTracker(num_players, trackers.WindowNodeConfig(window_size=int(6.0 * cam_settings.fps)))
        self.window_tracker_S =     trackers.AllWindowTracker(num_players, trackers.WindowNodeConfig(window_size=int(6.0 * cam_settings.fps)))
        self.window_tracker_I =     trackers.AllWindowTracker(num_players, trackers.WindowNodeConfig(window_size=int(6.0 * self.render_settings.window.avg_fps)))

        self.bbox_filters =      trackers.FilterTracker(
            num_players,
            [
                lambda: nodes.BBoxEuroSmoother(self.b_box_smooth_config),
                lambda: nodes.BBoxPredictor(pose_group.bbox.prediction),
                # lambda: nodes.BBoxRateLimiter(self.b_box_rate_config),
            ]
        )

        self.pose_raw_filters =     trackers.FilterTracker(
            num_players,
            [
                lambda: nodes.PointDualConfFilter(nodes.DualConfFilterConfig(threshold_low=pose_settings.confidence_low, threshold_high=pose_settings.confidence_high)),
                # lambda: nodes.PointTemporalStabilizer(nodes.TemporalStabilizerConfig()),
                nodes.AngleExtractor,
                lambda: nodes.AngleVelExtractor(fps=cam_settings.fps),
                # lambda: nodes.PoseValidator(nodes.ValidatorConfig(name="Raw")),
            ]
        )

        self.pose_smooth_filters = trackers.FilterTracker(
            num_players,
            [
                lambda: nodes.PointEuroSmoother(self.point_smooth_config),
                nodes.AngleExtractor,
                lambda: nodes.AngleVelExtractor(fps=cam_settings.fps),
                lambda: nodes.AngleVelEuroSmoother(self.angle_smooth_config),
                lambda: nodes.AngleEuroSmoother(self.angle_smooth_config),
                # lambda: nodes.AngleStickyFiller(nodes.StickyFillerConfig(init_to_zero=False, hold_scores=False)),
                lambda: nodes.AngleMotionExtractor(self.motion_extractor_config),
                lambda: nodes.AngleMotionMovingAverageSmoother(self.motion_ma_config),
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
            num_players,
            [
                lambda: nodes.PointPredictor(pose_group.point.prediction),
                lambda: nodes.AnglePredictor(pose_group.angle.prediction),
                lambda: nodes.AngleVelPredictor(pose_group.angle.prediction),
                lambda: nodes.AngleStickyFiller(nodes.StickyFillerConfig(init_to_zero=False, hold_scores=True)),
                lambda: nodes.SimilarityStickyFiller(nodes.StickyFillerConfig(init_to_zero=True, hold_scores=False)),
                # lambda: nodes.PoseValidator(nodes.ValidatorConfig(name="Prediction")),
            ]
        )

        self.interpolator = trackers.InterpolatorTracker(
            num_players,
            [
                lambda: nodes.BBoxChaseInterpolator(self.b_box_interp_config),
                lambda: nodes.PointChaseInterpolator(self.point_interp_config),
                lambda: nodes.AngleChaseInterpolator(self.angle_interp_config),
                lambda: nodes.AngleVelChaseInterpolator(self.angle_interp_config),
                lambda: nodes.SimilarityChaseInterpolator(self.simil_interp_config),
            ]
        )

        self.pose_interpolation_pipeline = trackers.FilterTracker(
            num_players,
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

        # CORRELATION COMPUTATION (alternative to similarity)
        self.window_tracker_S.add_callback(self.window_correlator.submit_all)
        self.window_correlator.add_callback(lambda result: self.similarity_applicator.submit(result[0]))
        self.window_correlator.add_callback(lambda result: self.leader_applicator.submit(result[1]))
        self.window_correlator.start()

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

        if self.player:
            self.player.start()
        if self.recorder:
            self.recorder.start() # start after gui to prevent record at startup

        self.is_running = True

        self.render.window_manager.add_exit_callback(self.stop)
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
        self.window_correlator.stop()

        self.settings_server.stop()


        for camera in self.cameras:
            camera.join(timeout=10)

        self.is_finished = True

