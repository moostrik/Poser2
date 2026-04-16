"""HD Trio — 3-camera interactive installation with fluid rendering."""

from typing import Optional
from functools import partial

from modules.oak import Camera, Simulator, Player, Sync, Recorder as VideoRecorder
from modules.settings import presets, NiceServer
from modules.data_hub import DataHub, Stage
from modules.inout import OscSound, ArtNetBars, OscReceiver
from modules.tracker import OnePerCamTracker
from modules.pose import batch, nodes, trackers, features, window
from modules.pose.recorder import Recorder as PoseRecorder
from modules.session import Session, Timeline
from modules.gl.WindowManager import WindowSettings

from .settings import HDTrioSettings
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
        p = self.settings.pose

        # DATA_HUB
        self.data_hub = DataHub()

        # SESSION
        self.session = Session(self.settings.session.core)
        self.session_osc = OscReceiver(self.settings.session.osc)
        self.session_osc.bind('/start/recording', self._on_osc_start_recording)
        self.session_osc.bind('/stop/recording',  self._on_osc_stop_recording)
        self.session_osc.bind('/group/id',        self._on_osc_group_id)
        self.timeline = Timeline(self.settings.session.timeline)
        self.video_recorder = VideoRecorder(self.settings.session.video)
        self.pose_recorder = PoseRecorder(self.settings.session.pose)

        self.timeline.add_stage_callback(lambda s: self.data_hub.set_timeline_stage(s))
        self.timeline.add_time_callback(lambda t: self.data_hub.set_timeline_stage_progress(self.settings.session.timeline.stage_progress))
        self.timeline.add_time_callback(lambda t: self.data_hub.set_timeline_progress(self.settings.session.timeline.progress))
        self.data_hub.add_update_callback(self.timeline.update)

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
        self.image_crop_processor = batch.ImageCropProcessor(p.image_crop)

        for camera in self.cameras:
            camera.add_preview_callback(self.data_hub.set_cam_frame)
            camera.add_sync_callback(self.video_recorder.set_synced_frames)
            camera.add_frame_callback(self.image_crop_processor.set_image)
            camera.add_frame_callback(self.frame_sync_bang.add_frame)
            camera.add_tracker_callback(self.tracker.add_cam_tracklets)
            camera.add_tracker_callback(self.data_hub.set_depth_tracklets)
            camera.add_tracker_callback(self.tracklet_sync_bang.add_frame)

        # DETECTION
        features.configure_features(num_players)

        self.poses_from_tracklets = batch.PosesFromTracklets(num_players)
        self.point_extractor = batch.PointBatchExtractor(p.detection)
        self.mask_extractor  = batch.MaskBatchExtractor(p.segmentation)
        self.flow_extractor  = batch.FlowBatchExtractor(p.flow)

        self.bbox_filters = trackers.FilterTracker({
            i: trackers.FilterPipeline([
                nodes.BBoxEuroSmoother(p.bbox.smoother),
                nodes.BBoxPredictor(p.bbox.prediction),
            ])
            for i in range(num_players)
        })

        self.tracker.add_tracklet_callback(self.poses_from_tracklets.submit_tracklets)
        self.tracker.add_tracklet_callback(self.data_hub.set_tracklets)
        self.tracklet_sync_bang.add_callback(self.tracker.notify_update)
        self.frame_sync_bang.add_callback(self.poses_from_tracklets.generate)

        self.poses_from_tracklets.add_frames_callback(self.bbox_filters.process)
        self.bbox_filters.add_frames_callback(self.image_crop_processor.process)
        self.image_crop_processor.add_callback(self.point_extractor.process)
        self.image_crop_processor.add_callback(self.mask_extractor.process)
        self.image_crop_processor.add_callback(self.flow_extractor.process)
        self.mask_extractor.add_callback(self.data_hub.set_gpu_frames)
        self.flow_extractor.add_callback(self.data_hub.set_flow_tensors)

        # STAGE RAW
        self.window_tracker_R = window.WindowTracker(num_players, p.window_raw)

        self.point_extractor.add_frames_callback(partial(self.data_hub.set_pose_frames, Stage.RAW))
        self.point_extractor.add_frames_callback(partial(self.pose_recorder.on_frame_dict, Stage.RAW))
        self.point_extractor.add_frames_callback(self.window_tracker_R.process)
        self.window_tracker_R.add_frame_windows_callback(partial(self.data_hub.set_pose_windows, Stage.RAW))

        # STAGE CLEAN
        self.pose_raw_filters = trackers.FilterTracker({
            i: trackers.FilterPipeline([
                nodes.PointDualConfFilter(p.point.confidence_filter),
                nodes.AngleExtractor(p.angle_extractor),
                nodes.AngleVelExtractor(p.velocity.extractor),
            ])
            for i in range(num_players)
        })
        self.window_tracker_C = window.WindowTracker(num_players, p.window_clean)

        self.point_extractor.add_frames_callback(self.pose_raw_filters.process)
        self.pose_raw_filters.add_frames_callback(partial(self.data_hub.set_pose_frames, Stage.CLEAN))
        self.pose_raw_filters.add_frames_callback(partial(self.pose_recorder.on_frame_dict, Stage.CLEAN))
        self.pose_raw_filters.add_frames_callback(self.window_tracker_C.process)
        self.window_tracker_C.add_frame_windows_callback(partial(self.data_hub.set_pose_windows, Stage.CLEAN))

        # STAGE SMOOTH
        self.similarity_applicator = nodes.SimilarityApplicator(p.similarity.similarity_applicator)
        self.leader_applicator     = nodes.LeaderScoreApplicator(p.similarity.leader_applicator)

        self.pose_smooth_filters = trackers.FilterTracker({
            i: trackers.FilterPipeline([
                nodes.PointEuroSmoother(p.point.smoother),
                nodes.AngleExtractor(p.angle_extractor),
                nodes.AngleVelExtractor(p.velocity.extractor),
                nodes.AngleVelEuroSmoother(p.velocity.smoother),
                nodes.AngleEuroSmoother(p.angle.smoother),
                nodes.AngleMotionExtractor(p.motion.extractor),
                nodes.AngleMotionMovingAverageSmoother(p.motion.moving_average),
                nodes.AngleSymExtractor(),
                nodes.MotionTimeExtractor(),
                nodes.AgeExtractor(),
                self.similarity_applicator,
                self.leader_applicator,
                nodes.SimilarityEuroSmoother(p.similarity.smoother),
            ])
            for i in range(num_players)
        })
        self.pose_prediction_filters = trackers.FilterTracker({
            i: trackers.FilterPipeline([
                nodes.PointPredictor(p.point.prediction),
                nodes.AnglePredictor(p.angle.prediction),
                nodes.AngleVelPredictor(p.velocity.prediction),
                nodes.AngleStickyFiller(p.angle.sticky),
                nodes.SimilarityStickyFiller(p.similarity.sticky),
            ])
            for i in range(num_players)
        })
        self.window_tracker_S = window.WindowTracker(num_players, p.window_smooth)

        self.pose_raw_filters.add_frames_callback(self.pose_smooth_filters.process)
        self.pose_smooth_filters.add_frames_callback(self.pose_prediction_filters.process)
        self.pose_prediction_filters.add_frames_callback(partial(self.data_hub.set_pose_frames, Stage.SMOOTH))
        self.pose_prediction_filters.add_frames_callback(partial(self.pose_recorder.on_frame_dict, Stage.SMOOTH))
        self.pose_smooth_filters.add_frames_callback(self.window_tracker_S.process)
        self.window_tracker_S.add_frame_windows_callback(partial(self.data_hub.set_pose_windows, Stage.SMOOTH))

        # STAGE LERP
        self.motion_gate_applicator = nodes.MotionGateApplicator(p.similarity.motion_gate)

        self.interpolators = trackers.InterpolatorTracker({
            i: trackers.InterpolatorPipeline([
                nodes.BBoxChaseInterpolator(p.bbox.interpolator),
                nodes.PointChaseInterpolator(p.point.interpolator),
                nodes.AngleChaseInterpolator(p.angle.interpolator),
                nodes.AngleVelChaseInterpolator(p.velocity.interpolator),
                nodes.SimilarityChaseInterpolator(p.similarity.interpolator),
            ])
            for i in range(num_players)
        })
        self.pose_interpolation_filters = trackers.FilterTracker({
            i: trackers.FilterPipeline([
                nodes.AngleSymExtractor(),
                nodes.MotionTimeExtractor(),
                nodes.AgeExtractor(),
                nodes.AngleVelStickyFiller(p.velocity.sticky),
                nodes.AngleVelEuroSmoother(p.velocity.smoother),
                nodes.AngleMotionExtractor(p.motion.extractor),
                nodes.AngleMotionMovingAverageSmoother(p.motion.moving_average),
            ])
            for i in range(num_players)
        })
        self.motion_gate_tracker = trackers.FilterTracker({
            i: trackers.FilterPipeline([self.motion_gate_applicator])
            for i in range(num_players)
        })
        self.window_tracker_I = window.WindowTracker(num_players, p.window_lerp)

        self.data_hub.add_update_callback(self.interpolators.update)
        self.pose_prediction_filters.add_frames_callback(self.interpolators.submit)
        self.interpolators.add_frames_callback(self.pose_interpolation_filters.process)
        self.pose_interpolation_filters.add_frames_callback(self.motion_gate_applicator.submit)
        self.pose_interpolation_filters.add_frames_callback(self.motion_gate_tracker.process)
        self.motion_gate_tracker.add_frames_callback(partial(self.data_hub.set_pose_frames, Stage.LERP))
        self.motion_gate_tracker.add_frames_callback(partial(self.pose_recorder.on_frame_dict, Stage.LERP))
        self.motion_gate_tracker.add_frames_callback(self.window_tracker_I.process)
        self.window_tracker_I.add_frame_windows_callback(partial(self.data_hub.set_pose_windows, Stage.LERP))

        # SIMILARITY
        self.window_similator  = batch.WindowSimilarity(p.similarity.window_similarity)
        self.window_correlator = batch.WindowCorrelation(p.similarity.window_correlation)

        self.window_tracker_S.add_frame_windows_callback(self.window_similator.submit_all)
        self.window_similator.add_callback(lambda result: self.similarity_applicator.submit(result[0]))
        self.window_similator.add_callback(lambda result: self.leader_applicator.submit(result[1]))

        self.window_tracker_S.add_frame_windows_callback(self.window_correlator.submit_all)
        self.window_correlator.add_callback(lambda result: self.similarity_applicator.submit(result[0]))
        self.window_correlator.add_callback(lambda result: self.leader_applicator.submit(result[1]))

        # RENDER
        self.render = HDTrioRender(self.data_hub, self.settings.render, self.settings.session.timeline)
        self.settings.render.window.bind(WindowSettings.avg_fps, self._on_render_fps)
        self.render.window_manager.add_exit_callback(self.stop)

        # IN/OUT
        self.sound_osc = OscSound(self.data_hub, self.settings.inout.osc_sound)
        self.artnet_controllers: list[ArtNetBars] = []
        for i in range(num_players):
            self.artnet_controllers.append(ArtNetBars(self.settings.inout.artnets[i]))

        self.data_hub.add_update_callback(self.sound_osc.notify_update)

    def _on_osc_start_recording(self, *_) -> None:
        self.settings.session.start = True

    def _on_osc_stop_recording(self, *_) -> None:
        self.settings.session.stop = True

    def _on_osc_group_id(self, gid: str, *_) -> None:
        self.settings.session.name = gid

    def _on_render_fps(self, fps: int) -> None:
        if fps > 0:
            self.settings.render_fps = float(fps)

    def start(self) -> None:
        self.settings_server.start()

        for camera in self.cameras:
            camera.start()

        self.tracker.start()
        self.point_extractor.start()
        self.mask_extractor.start()
        self.flow_extractor.start()
        self.window_similator.start()
        self.window_correlator.start()

        self.sound_osc.start()
        for artnet in self.artnet_controllers:
            artnet.start()

        if self.player:
            self.player.start()
        self.video_recorder.start()

        self.is_running = True
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
