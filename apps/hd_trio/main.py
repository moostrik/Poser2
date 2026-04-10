"""HD Trio — 3-camera interactive installation with fluid rendering."""

from typing import Optional
from functools import partial

from modules.oak import Camera, Simulator, Player, Sync, Recorder as VideoRecorder
from modules.settings import presets, NiceServer
from modules.data_hub import DataHub, Stage
from modules.inout import OscSound, ArtNetBars, OscReceiver
from modules.tracker import OnePerCamTracker
from modules.pose import batch, nodes, trackers, features
# from modules.pose.features import configure_features
from modules.pose.recorder import Recorder as PoseRecorder
from modules.session import Session
from modules.utils import Timeline
from modules.gl.WindowManager import WindowSettings

from .settings import HDTrioSettings, SHOW_STAGE_DURATIONS
from .render import HDTrioRender

APP_NAME = 'hd_trio'


class HDTrioMain:
    def __init__(self, simulation: bool = False) -> None:

        self.is_running: bool = False
        self.is_finished: bool = False

        # SETTINGS
        presets.set_app(APP_NAME)
        self.settings = HDTrioSettings()
        preset_file = presets.startup_path()
        if not presets.load(self.settings, preset_file):
            raise FileNotFoundError(f"No preset found for '{APP_NAME}' at {preset_file}")
        self.settings.camera.sim_enabled = simulation
        self.settings.initialize()
        self.settings_server = NiceServer(self.settings, self.settings.server, on_exit=self.stop)

        num_players: int = self.settings.num_players

        # DATA_HUB
        self.data_hub = DataHub()

        # SESSION (osc, timeline, video & pose recorders)
        self.session = Session(self.settings.session)
        self.session_osc = OscReceiver(self.settings.session.osc)
        self.session_osc.bind('/start/recording', self._on_osc_start_recording)
        self.session_osc.bind('/stop/recording',  self._on_osc_stop_recording)
        self.session_osc.bind('/group/id',        self._on_osc_group_id)
        self.timeline = Timeline(self.settings.session.timeline, SHOW_STAGE_DURATIONS)
        self.video_recorder = VideoRecorder(self.settings.session.video)
        self.pose_recorder = PoseRecorder(self.settings.session.pose)

        # CAMERA
        self.cameras: list[Camera | Simulator] = []
        self.player: Optional[Player] = None
        if self.settings.camera.sim_enabled:
            self.player = Player(self.settings.camera.simulator)
            for i in range(num_players):
                self.cameras.append(Simulator(self.player, self.settings.camera.cameras[i], self.settings.camera.simulator))
        else:
            for i in range(num_players):
                self.cameras.append(Camera(self.settings.camera.cameras[i]))
        self.frame_sync_bang = Sync(self.settings.camera.frame_sync, False, 'frame_sync')
        self.tracker = OnePerCamTracker(self.settings.camera.tracker, num_players)
        self.tracklet_sync_bang = Sync(self.settings.camera.tracklet_sync, False, 'tracklet_sync')

        # RENDER
        self.render = HDTrioRender(self.data_hub, self.settings.render, num_cams=len(self.cameras), num_players=num_players)

        # IN_OUT
        self.sound_osc = OscSound(self.data_hub, self.settings.inout.osc_sound)
        self.artnet_controllers: list[ArtNetBars] = []
        for i in range(num_players):
            self.artnet_controllers.append(ArtNetBars(self.settings.inout.artnets[i]))

        # POSE PROCESSING PIPELINES
        features.configure_features(num_players)

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
            ]
        )

        self.pose_raw_filters =     trackers.FilterTracker(
            num_players,
            [
                lambda: nodes.PointDualConfFilter(self.settings.pose.point.confidence_filter),
                lambda: nodes.AngleExtractor(nodes.AngleExtractorSettings(aspect_ratio=self.settings.pose.detection.aspect_ratio)),
                lambda: nodes.AngleVelExtractor(self.settings.pose.velocity.extractor),
            ]
        )

        self.pose_smooth_filters = trackers.FilterTracker(
            num_players,
            [
                lambda: nodes.PointEuroSmoother(self.settings.pose.point.smoother),
                lambda: nodes.AngleExtractor(nodes.AngleExtractorSettings(aspect_ratio=self.settings.pose.detection.aspect_ratio)),
                lambda: nodes.AngleVelExtractor(self.settings.pose.velocity.extractor),
                lambda: nodes.AngleVelEuroSmoother(self.settings.pose.velocity.smoother),
                lambda: nodes.AngleEuroSmoother(self.settings.pose.angle.smoother),
                lambda: nodes.AngleMotionExtractor(self.settings.pose.motion.extractor),
                lambda: nodes.AngleMotionMovingAverageSmoother(self.settings.pose.motion.moving_average),
                nodes.AngleSymExtractor,
                nodes.MotionTimeExtractor,
                nodes.AgeExtractor,
                lambda: self.similarity_applicator,
                lambda: self.leader_applicator,
                lambda: nodes.SimilarityEuroSmoother(self.settings.pose.similarity.smoother),
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
                nodes.AngleSymExtractor,
                nodes.MotionTimeExtractor,
                nodes.AgeExtractor,
                lambda: nodes.AngleVelStickyFiller(self.settings.pose.velocity.sticky),
                lambda: nodes.AngleVelEuroSmoother(self.settings.pose.velocity.smoother),
                lambda: nodes.AngleMotionExtractor(self.settings.pose.motion.extractor),
                lambda: nodes.AngleMotionMovingAverageSmoother(self.settings.pose.motion.moving_average),
            ]
        )

    def _on_osc_start_recording(self, *_) -> None:
        self.settings.session.record = True

    def _on_osc_stop_recording(self, *_) -> None:
        self.settings.session.record = False

    def _on_osc_group_id(self, gid: str, *_) -> None:
        self.settings.session.group_id = gid

    def start(self) -> None:

        self.settings_server.start()

        def _update_render_fps(fps: int) -> None:
            if fps > 0:
                self.settings.render_fps = float(fps)

        self.settings.render.window.bind(WindowSettings.avg_fps, _update_render_fps)

        for camera in self.cameras:
            camera.add_preview_callback(self.data_hub.set_cam_frame)
            if not self.settings.camera.sim_enabled:
                camera.add_sync_callback(self.video_recorder.set_synced_frames)
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
        self.point_extractor.add_frames_callback(self.pose_recorder.on_frame_dict)
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
        self.pose_interpolation_filters.add_frames_callback(self.motion_gate_applicator.submit)
        self.pose_interpolation_filters.add_frames_callback(self.motion_gate_tracker.process)
        self.motion_gate_tracker.add_frames_callback(partial(self.data_hub.set_pose_frames, Stage.LERP))
        self.motion_gate_tracker.add_frames_callback(self.window_tracker_I.process)
        self.window_tracker_I.add_frame_windows_callback(partial(self.data_hub.set_pose_windows, Stage.LERP))

        # SIMILARITY COMPUTATION
        self.window_tracker_S.add_frame_windows_callback(self.window_similator.submit_all)
        self.window_similator.add_callback(lambda result: self.similarity_applicator.submit(result[0]))
        self.window_similator.add_callback(lambda result: self.leader_applicator.submit(result[1]))
        self.window_similator.start()

        # CORRELATION COMPUTATION
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

        # TIMELINE (ticked by render loop via data_hub.notify_update)
        self.timeline.add_stage_callback(lambda s: self.data_hub.set_timeline_stage(s))
        self.timeline.add_time_callback(lambda t: self.data_hub.set_timeline_stage_progress(self.settings.session.timeline.stage_progress))
        self.timeline.add_time_callback(lambda t: self.data_hub.set_timeline_progress(self.settings.session.timeline.progress))
        self.data_hub.add_update_callback(self.timeline.update)

        if self.player:
            self.player.start()
        self.video_recorder.start()

        self.is_running = True

        self.render.window_manager.add_exit_callback(self.stop)
        self.render.window_manager.start()

    def stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False

        self.settings_server.stop()

        self.render.window_manager.stop()

        if self.player:
            self.player.stop()
        for camera in self.cameras:
            camera.stop()
        self.video_recorder.stop()

        self.tracker.stop()
        self.sound_osc.stop()

        for artnet in self.artnet_controllers:
            artnet.stop()

        if self.session_osc:
            self.session_osc.server.shutdown()

        self.point_extractor.stop()
        self.mask_extractor.stop()
        self.flow_extractor.stop()

        self.window_similator.stop()
        self.window_correlator.stop()

        for camera in self.cameras:
            camera.join(timeout=10)

        self.is_finished = True
