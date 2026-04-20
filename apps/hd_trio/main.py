"""HD Trio — 3-camera interactive installation with fluid rendering."""

from typing import Optional
from functools import partial

from modules.utils import Broadcast
from modules.oak import Camera, Simulator, Player, Sync, Recorder as VideoRecorder
from modules.settings import presets, NiceServer
from modules.inout import OscSound, ArtNetBars, OscReceiver
from modules.tracker import OnePerCamTracker
from modules.pose import batch, nodes, trackers, features, window
from modules.pose.recorder import Recorder as PoseRecorder
from modules.session import Session, Sequencer
from modules.gl.WindowManager import WindowSettings

from .render_board import RenderBoard
from .settings import Settings, Stage
from .render import HDTrioRender

APP_NAME = 'hd_trio'


class HDTrioMain:
    def __init__(self, simulation: bool = False) -> None:

        self.is_running: bool = False
        self.is_finished: bool = False

        # SETTINGS
        presets.set_app(APP_NAME)
        self.settings = Settings()
        preset_file = presets.startup_path()
        if not presets.load(self.settings, preset_file):
            raise FileNotFoundError(f"No preset found for '{APP_NAME}' at {preset_file}")
        self.settings.camera.sim_enabled = simulation
        self.settings.initialize()
        self.settings_server = NiceServer(self.settings, self.settings.server, on_exit=self.stop)

        num_players: int = self.settings.num_players
        ps = self.settings.pose

        # BLACKBOARD
        self.board = RenderBoard()

        # SESSION
        self.session = Session(self.settings.session.core)
        self.session_osc = OscReceiver(self.settings.session.osc)
        self.session_osc.bind('/start/recording', self._on_osc_start_recording)
        self.session_osc.bind('/stop/recording',  self._on_osc_stop_recording)
        self.session_osc.bind('/group/id',        self._on_osc_group_id)
        self.sound_osc = OscSound(self.settings.inout.osc_sound)
        self.sequencer = Sequencer(self.settings.session.sequencer)
        self.sequencer.add_state_callback(self.board.set_sequence)
        self.sequencer.add_state_callback(self.sound_osc.set_sequencer_state)
        self.video_recorder = VideoRecorder(self.settings.session.video)
        self.pose_recorder = PoseRecorder(self.settings.session.pose)
        self.artnet_controllers: list[ArtNetBars] = []
        for i in range(num_players):
            self.artnet_controllers.append(ArtNetBars(self.settings.inout.artnets[i]))

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
        self.image_crop_processor = batch.ImageCropProcessor(ps.image_crop)

        for camera in self.cameras:
            camera.add_sync_callback(self.video_recorder.submit_synced_frames)
            camera.add_frame_callback(self.image_crop_processor.set_image)
            camera.add_frame_callback(self.frame_sync_bang.submit_frame)
            camera.add_tracker_callback(self.tracker.submit_cam_tracklets)
            camera.add_tracker_callback(self.board.set_depth_tracklets)
            camera.add_tracker_callback(self.tracklet_sync_bang.submit_frame)

        # DETECTION
        features.configure_features(num_players)

        self.poses_from_tracklets = batch.PosesFromTracklets(num_players)
        self.point_extractor = batch.PointBatchExtractor(ps.detection)
        self.mask_extractor  = batch.MaskBatchExtractor(ps.segmentation)

        self.bbox_filters = trackers.FilterTracker({
            i: trackers.FilterPipeline([
                nodes.BBoxEuroSmoother(ps.bbox.smoother),
                nodes.BBoxPredictor(ps.bbox.prediction),
            ])
            for i in range(num_players)
        })

        self.tracker.add_tracklet_callback(self.poses_from_tracklets.set_tracklets)
        self.tracklet_sync_bang.add_sync_callback(self.tracker.notify_update)
        self.frame_sync_bang.add_sync_callback(self.poses_from_tracklets.process)

        self.poses_from_tracklets.add_frames_callback(self.bbox_filters.process)
        self.bbox_filters.add_frames_callback(self.image_crop_processor.process)
        self.image_crop_processor.add_image_callback(self.point_extractor.process)
        self.image_crop_processor.add_image_callback(self.mask_extractor.process)
        self.mask_extractor.add_image_callback(lambda _f, gpu: self.board.set_images(gpu))

        # STAGE WINDOW TRACKERS & BROADCASTS
        self.window_trackers: dict[Stage, window.WindowTracker] = {}
        self.stages: dict[Stage, Broadcast] = {}
        for stage in Stage:
            wt = window.WindowTracker(num_players, getattr(ps, f'window_{stage.name.lower()}'))
            wt.add_windows_callback(partial(self.board.set_windows, stage))
            self.window_trackers[stage] = wt
            self.stages[stage] = Broadcast([
                partial(self.board.set_frames, stage),
                partial(self.pose_recorder.submit_frames, stage),
                partial(self.sound_osc.set_frames, stage),
                wt.process,
            ])

        # POSE STAGE RAW
        self.point_extractor.add_frames_callback(self.stages[Stage.RAW])

        # POSE STAGE CLEAN
        self.filters_clean = trackers.FilterTracker({
            i: trackers.FilterPipeline([
                nodes.PointDualConfFilter(ps.point.confidence_filter),
                nodes.AngleExtractor(ps.angle_extractor),
                nodes.AngleVelExtractor(ps.velocity.extractor),
            ])
            for i in range(num_players)
        })
        self.stages[Stage.RAW].add_callback(self.filters_clean.process)
        self.filters_clean.add_frames_callback(self.stages[Stage.CLEAN])

        # POSE STAGE SMOOTH
        self.similarity_applicator = nodes.SimilarityApplicator(ps.similarity.similarity_applicator)
        self.leader_applicator     = nodes.LeaderScoreApplicator(ps.similarity.leader_applicator)

        self.filters_smooth = trackers.FilterTracker({
            i: trackers.FilterPipeline([
                nodes.PointEuroSmoother(ps.point.smoother),
                nodes.AngleExtractor(ps.angle_extractor),
                nodes.AngleVelExtractor(ps.velocity.extractor),
                nodes.AngleVelEuroSmoother(ps.velocity.smoother),
                nodes.AngleEuroSmoother(ps.angle.smoother),
                nodes.AngleMotionExtractor(ps.motion.extractor),
                nodes.AngleMotionMovingAverageSmoother(ps.motion.moving_average),
                nodes.AngleSymExtractor(),
                nodes.MotionTimeExtractor(),
                nodes.AgeExtractor(),
                self.similarity_applicator,
                self.leader_applicator,
                nodes.SimilarityEuroSmoother(ps.similarity.smoother),
            ])
            for i in range(num_players)
        })
        self.stages[Stage.CLEAN].add_callback(self.filters_smooth.process)
        self.filters_smooth.add_frames_callback(self.stages[Stage.SMOOTH])

        self.window_similator  = batch.WindowSimilarity(ps.similarity.window_similarity)
        self.window_correlator = batch.WindowCorrelation(ps.similarity.window_correlation)

        self.window_trackers[Stage.SMOOTH].add_windows_callback(self.window_similator.submit)
        self.window_similator.add_similarity_callback(self.similarity_applicator.set)
        self.window_similator.add_similarity_callback(self.leader_applicator.set)

        self.window_trackers[Stage.SMOOTH].add_windows_callback(self.window_correlator.submit)
        self.window_correlator.add_similarity_callback(self.similarity_applicator.set)
        self.window_correlator.add_similarity_callback(self.leader_applicator.set)

        # POSE STAGE PREDICT
        self.filters_predict = trackers.FilterTracker({
            i: trackers.FilterPipeline([
                nodes.PointPredictor(ps.point.prediction),
                nodes.AnglePredictor(ps.angle.prediction),
                nodes.AngleVelPredictor(ps.velocity.prediction),
                nodes.AngleStickyFiller(ps.angle.sticky),
                nodes.SimilarityStickyFiller(ps.similarity.sticky),
            ])
            for i in range(num_players)
        })
        self.stages[Stage.SMOOTH].add_callback(self.filters_predict.process)
        self.filters_predict.add_frames_callback(self.stages[Stage.PREDICT])

        # POSE STAGE LERP
        self.motion_gate_applicator = nodes.MotionGateApplicator(ps.similarity.motion_gate)

        self.interpolators_lerp = trackers.InterpolatorTracker({
            i: trackers.InterpolatorPipeline([
                nodes.BBoxChaseInterpolator(ps.bbox.interpolator),
                nodes.PointChaseInterpolator(ps.point.interpolator),
                nodes.AngleChaseInterpolator(ps.angle.interpolator),
                nodes.AngleVelChaseInterpolator(ps.velocity.interpolator),
                nodes.SimilarityChaseInterpolator(ps.similarity.interpolator),
            ])
            for i in range(num_players)
        })
        self.filters_lerp = trackers.FilterTracker({
            i: trackers.FilterPipeline([
                nodes.AngleSymExtractor(),
                nodes.MotionTimeExtractor(),
                nodes.AgeExtractor(),
                nodes.AngleVelStickyFiller(ps.velocity.sticky),
                nodes.AngleVelEuroSmoother(ps.velocity.smoother),
                nodes.AngleMotionExtractor(ps.motion.extractor),
                nodes.AngleMotionMovingAverageSmoother(ps.motion.moving_average),
            ])
            for i in range(num_players)
        })
        self.gate_lerp = trackers.FilterTracker({
            i: trackers.FilterPipeline([self.motion_gate_applicator])
            for i in range(num_players)
        })
        self.stages[Stage.PREDICT].add_callback(self.interpolators_lerp.set)
        self.interpolators_lerp.add_frames_callback(self.filters_lerp.process)
        self.filters_lerp.add_frames_callback(self.motion_gate_applicator.set)
        self.filters_lerp.add_frames_callback(self.gate_lerp.process)
        self.gate_lerp.add_frames_callback(self.stages[Stage.LERP])

        # RENDER
        self.render = HDTrioRender(self.board, self.settings.render)
        self.settings.render.window.bind(WindowSettings.avg_fps, self._on_render_fps)
        self.render.add_update_callback(self.sequencer.update)
        self.render.add_update_callback(self.interpolators_lerp.update)
        self.render.add_exit_callback(self.stop)

    def start(self) -> None:
        self.settings_server.start()

        for camera in self.cameras:
            camera.start()

        self.tracker.start()
        self.point_extractor.start()
        self.mask_extractor.start()
        self.window_similator.start()
        self.window_correlator.start()

        self.sound_osc.start()
        for artnet in self.artnet_controllers:
            artnet.start()

        if self.player:
            self.player.start()
        self.video_recorder.start()

        self.is_running = True
        self.render.start()

    def stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False

        self.settings_server.stop()

        self.render.stop()

        if self.player:
            self.player.stop()
        for camera in self.cameras:
            camera.stop()
        self.video_recorder.stop()

        self.tracker.stop()
        self.sound_osc.stop()

        for artnet in self.artnet_controllers:
            artnet.stop()

        self.session_osc.server.shutdown()

        self.point_extractor.stop()
        self.mask_extractor.stop()
        self.window_similator.stop()
        self.window_correlator.stop()

        for camera in self.cameras:
            camera.join(timeout=10)

        self.is_finished = True

    def _on_osc_start_recording(self, *_) -> None:
        self.settings.session.start = True

    def _on_osc_stop_recording(self, *_) -> None:
        self.settings.session.stop = True

    def _on_osc_group_id(self, gid: str, *_) -> None:
        self.settings.session.name = gid

    def _on_render_fps(self, fps: int) -> None:
        if fps > 0:
            self.settings.render_fps = float(fps)
