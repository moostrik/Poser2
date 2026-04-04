# Standard library imports
from typing import Optional
from functools import partial

# Local application imports
from modules.main_settings import MainSettings
from modules.oak import Camera, Simulator, Player, Recorder, Sync
from modules.settings import presets, NiceServer
from modules.render import RenderManager
from modules.data_hub import DataHub, Stage
from modules.inout import OscSound, ArtNetBars
from modules.tracker import OnePerCamTracker
from modules.pose import batch, nodes, trackers
from modules.utils import Timer


class Main():
    def __init__(self, simulation: bool = False) -> None:

        self.is_running: bool = False
        self.is_finished: bool = False

        # SETTINGS
        self.settings = MainSettings()
        if not presets.load(self.settings, presets.startup_path()):
            presets.save(self.settings, presets.startup_path())  # create default preset if loading failed (not responsibility of main!)
        self.settings.camera.sim_enabled = simulation
        self.settings.initialize()
        self.settings_server = NiceServer(self.settings, self.settings.server, on_exit=self.stop)

        num_players: int = self.settings.num_players

        # DATA_HUB
        self.data_hub = DataHub()

        # CAMERA
        self.cameras: list[Camera | Simulator] = []
        self.recorder: Optional[Recorder] = None
        self.player: Optional[Player] = None
        if self.settings.camera.sim_enabled:
            self.player = Player(self.settings.camera.simulator)
            for i in range(num_players):
                self.cameras.append(Simulator(self.player, self.settings.camera.cameras[i], self.settings.camera.simulator))
        else:
            self.recorder = Recorder(self.settings.camera.recorder)
            for i in range(num_players):
                camera = Camera(self.settings.camera.cameras[i])
                self.cameras.append(camera)
        self.frame_sync_bang = Sync(self.settings.camera.frame_sync, False, 'frame_sync')

        # TRACKER
        self.tracker = OnePerCamTracker(self.settings.tt.tracker, num_players)
        self.tracklet_sync_bang = Sync(self.settings.camera.tracklet_sync, False, 'tracklet_sync')

        # TIMER
        self.timer = Timer(self.settings.tt.timer)
        self.render = RenderManager(self.data_hub, self.settings.render, num_cams=len(self.cameras), num_players=num_players)

        # IN_OUT
        self.sound_osc = OscSound(self.data_hub, self.settings.inout.osc_sound)
        self.artnet_controllers: list[ArtNetBars] = []
        for i in range(num_players):
            self.artnet_controllers.append(ArtNetBars(self.settings.inout.artnets[i]))

        # POSE PROCESSING PIPELINES
        self.poses_from_tracklets = batch.PosesFromTracklets(num_players)

        self.image_crop_processor = batch.ImageCropProcessor(self.settings.pose.image_crop)
        self.point_extractor =      batch.PointBatchExtractor(self.settings.pose.detection)
        self.mask_extractor =       batch.MaskBatchExtractor(self.settings.pose.segmentation)
        self.flow_extractor =       batch.FlowBatchExtractor(self.settings.pose.flow)

        self.window_similator =     batch.WindowSimilarity(self.settings.pose.similarity.window_similarity)
        self.window_correlator =    batch.WindowCorrelation(self.settings.pose.similarity.window_correlation)

        # Feature applicators
        self.similarity_applicator =    nodes.SimilarityApplicator(self.settings.pose.similarity.similarity_applicator)
        self.leader_applicator =        nodes.LeaderScoreApplicator(self.settings.pose.similarity.leader_applicator)
        self.motion_gate_applicator =   nodes.MotionGateApplicator(self.settings.pose.similarity.motion_gate)
        self.motion_gate_tracker =      trackers.FilterTracker(num_players, [lambda: self.motion_gate_applicator])

        # WINDOW TRACKERS
        self.window_tracker_R =     trackers.FrameWindowTracker(num_players, self.settings.pose.window_raw)
        self.window_tracker_S =     trackers.FrameWindowTracker(num_players, self.settings.pose.window_smooth)
        self.window_tracker_I =     trackers.FrameWindowTracker(num_players, self.settings.pose.window_lerp)

        self.bbox_filters =      trackers.FilterTracker(
            num_players,
            [
                lambda: nodes.BBoxEuroSmoother(self.settings.pose.bbox.smoother),
                lambda: nodes.BBoxPredictor(self.settings.pose.bbox.prediction),
                # lambda: nodes.BBoxRateLimiter(self.new_settings.pose.rate_limiter),
            ]
        )

        self.pose_raw_filters =     trackers.FilterTracker(
            num_players,
            [
                lambda: nodes.PointDualConfFilter(self.settings.pose.point.confidence_filter),
                # lambda: nodes.PointTemporalStabilizer(nodes.TemporalStabilizerSettings()),
                nodes.AngleExtractor,
                lambda: nodes.AngleVelExtractor(self.settings.pose.velocity.extractor),
                # lambda: nodes.PoseValidator(nodes.ValidatorSettings(name="Raw")),
            ]
        )

        self.pose_smooth_filters = trackers.FilterTracker(
            num_players,
            [
                lambda: nodes.PointEuroSmoother(self.settings.pose.point.smoother),
                nodes.AngleExtractor,
                lambda: nodes.AngleVelExtractor(self.settings.pose.velocity.extractor),
                lambda: nodes.AngleVelEuroSmoother(self.settings.pose.velocity.smoother),
                lambda: nodes.AngleEuroSmoother(self.settings.pose.angle.smoother),
                # lambda: nodes.AngleStickyFiller(nodes.StickyFillerSettings(init_to_zero=False, hold_scores=False)),
                lambda: nodes.AngleMotionExtractor(self.settings.pose.motion.extractor),
                lambda: nodes.AngleMotionMovingAverageSmoother(self.settings.pose.motion.moving_average),
                # lambda: nodes.AngleMotionEasingNode(self.new_settings.pose.easing),
                nodes.AngleSymExtractor,
                nodes.MotionTimeExtractor,
                nodes.AgeExtractor,
                lambda: self.similarity_applicator,
                lambda: self.leader_applicator,
                lambda: nodes.SimilarityEuroSmoother(self.settings.pose.similarity.smoother),
                # lambda: nodes.PoseValidator(nodes.ValidatorSettings(name="Smooth")),
            ]
        )

        self.pose_prediction_filters = trackers.FilterTracker(
            num_players,
            [
                lambda: nodes.PointPredictor(self.settings.pose.point.prediction),
                lambda: nodes.AnglePredictor(self.settings.pose.angle.prediction),
                lambda: nodes.AngleVelPredictor(self.settings.pose.velocity.prediction),
                lambda: nodes.AngleStickyFiller(self.settings.pose.angle.sticky),
                lambda: nodes.SimilarityStickyFiller(self.settings.pose.similarity.sticky),
                # lambda: nodes.PoseValidator(nodes.ValidatorSettings(name="Prediction")),
            ]
        )

        self.interpolators = trackers.InterpolatorTracker(
            num_players,
            [
                lambda: nodes.BBoxChaseInterpolator(self.settings.pose.bbox.interpolator),
                lambda: nodes.PointChaseInterpolator(self.settings.pose.point.interpolator),
                lambda: nodes.AngleChaseInterpolator(self.settings.pose.angle.interpolator),
                lambda: nodes.AngleVelChaseInterpolator(self.settings.pose.velocity.interpolator),
                lambda: nodes.SimilarityChaseInterpolator(self.settings.pose.similarity.interpolator),
            ]
        )

        self.pose_interpolation_filters = trackers.FilterTracker(
            num_players,
            [
                # lambda: nodes.AngleVelExtractor(fps=settings.render.fps),
                nodes.AngleSymExtractor,
                nodes.MotionTimeExtractor,
                nodes.AgeExtractor,
                lambda: nodes.AngleVelStickyFiller(self.settings.pose.velocity.sticky),
                lambda: nodes.AngleVelEuroSmoother(self.settings.pose.velocity.smoother),
                lambda: nodes.AngleMotionExtractor(self.settings.pose.motion.extractor),
                lambda: nodes.AngleMotionMovingAverageSmoother(self.settings.pose.motion.moving_average),
                # lambda: nodes.PoseValidator(nodes.ValidatorSettings(name="Interpolation")),
            ]
        )

    def start(self) -> None:

        self.settings_server.start()

        for camera in self.cameras:
            camera.add_preview_callback(self.data_hub.set_cam_frame)
            if self.recorder:
                camera.add_sync_callback(self.recorder.set_synced_frames)
            camera.add_frame_callback(self.image_crop_processor.set_image)
            camera.add_frame_callback(self.frame_sync_bang.add_frame)
            camera.add_tracker_callback(self.tracker.add_cam_tracklets)
            camera.add_tracker_callback(self.data_hub.set_depth_tracklets)
            camera.add_tracker_callback(self.tracklet_sync_bang.add_frame)
            camera.start()

        # BBOX
        self.poses_from_tracklets.add_frames_callback(self.bbox_filters.process)
        self.bbox_filters.add_frames_callback(self.image_crop_processor.process)

        # POSE RAW
        self.point_extractor.add_frames_callback(self.pose_raw_filters.process)
        self.pose_raw_filters.add_frames_callback(partial(self.data_hub.set_pose_frames, Stage.RAW))
        self.pose_raw_filters.add_frames_callback(self.window_tracker_R.process)
        self.window_tracker_R.add_frame_windows_callback(partial(self.data_hub.set_pose_windows, Stage.RAW))

        # POSE SMOOTH & PREDICT
        self.pose_raw_filters.add_frames_callback(self.pose_smooth_filters.process)
        self.pose_smooth_filters.add_frames_callback(self.pose_prediction_filters.process)
        self.pose_prediction_filters.add_frames_callback(partial(self.data_hub.set_pose_frames, Stage.SMOOTH))
        self.pose_smooth_filters.add_frames_callback(self.window_tracker_S.process)
        self.window_tracker_S.add_frame_windows_callback(partial(self.data_hub.set_pose_windows, Stage.SMOOTH))

        # INTERPOLATION
        self.data_hub.add_update_callback(self.interpolators.update)
        self.pose_prediction_filters.add_frames_callback(self.interpolators.submit)
        self.interpolators.add_frames_callback(self.pose_interpolation_filters.process)

        # MOTION GATE (after interpolation pipeline, before DataHub)
        self.pose_interpolation_filters.add_frames_callback(self.motion_gate_applicator.submit) # dit slaat nergens op??
        self.pose_interpolation_filters.add_frames_callback(self.motion_gate_tracker.process)
        self.motion_gate_tracker.add_frames_callback(partial(self.data_hub.set_pose_frames, Stage.LERP))
        self.motion_gate_tracker.add_frames_callback(self.window_tracker_I.process)
        self.window_tracker_I.add_frame_windows_callback(partial(self.data_hub.set_pose_windows, Stage.LERP))

        # SIMILARITY COMPUTATION (uses combined callback for motion gate and velocity weighting)
        self.window_tracker_S.add_frame_windows_callback(self.window_similator.submit_all)
        self.window_similator.add_callback(lambda result: self.similarity_applicator.submit(result[0]))
        self.window_similator.add_callback(lambda result: self.leader_applicator.submit(result[1]))
        self.window_similator.start()

        # CORRELATION COMPUTATION (alternative to similarity)
        self.window_tracker_S.add_frame_windows_callback(self.window_correlator.submit_all)
        self.window_correlator.add_callback(lambda result: self.similarity_applicator.submit(result[0]))
        self.window_correlator.add_callback(lambda result: self.leader_applicator.submit(result[1]))
        self.window_correlator.start()

        # POSE ESTIMATION
        self.image_crop_processor.add_callback(self.point_extractor.process)
        self.point_extractor.start()

        # SEGMENTATION
        self.image_crop_processor.add_callback(self.mask_extractor.process)
        self.mask_extractor.add_callback(self.data_hub.set_gpu_frames)
        self.mask_extractor.start()

        # FLOW
        self.image_crop_processor.add_callback(self.flow_extractor.process)
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

