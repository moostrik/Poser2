"""White Space — 3-camera panoramic installation with circular LED light output."""

from typing import Optional
from functools import partial

import numpy as np

from modules.utils import Broadcast
from modules.oak import Camera, Simulator, Player, Sync, Recorder as VideoRecorder, FrameType
from modules.settings import presets, NiceServer
from modules.inout import OscReceiver
from modules.tracker import PanoramicTracker, PosesFromTracklets
from modules.pose import nodes, trackers, features, window, analytics, FrameDict
from modules.inference import source, crop, pose, segmentation
from modules.session import Session, Sequencer
from modules.gl import WindowSettings

from .board import Board
from .pose import PlayheadOffset, PlayheadOffsetExtractor, PlayheadStability, PlayheadStabilityExtractor, Ghoster, GhostedFeature
from .light import Render as LightRender
from .inout import OscLight, OscSound, UdpReceiver
from .render import Render as WindowRender
from .settings import Settings, Stage

APP_NAME = 'white_space'
DATA_PATH = 'apps/white_space/data'

import logging
logger = logging.getLogger(__name__)


class WhiteSpaceMain:
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
        num_cameras: int = self.settings.camera.num_cameras
        logging.info("Settings loaded: %s players, %s cameras, simulation=%s", num_players, num_cameras, simulation)
        ps = self.settings.pose

        # BLACKBOARD
        self.board = Board()

        # SESSION
        self.session = Session(self.settings.session.core)
        self.osc_sound = OscSound(self.settings.inout.osc_sound)
        self.ghoster = Ghoster(ps.ghoster, playhead=self.board.get_playhead)   # live/pool counts shared from root
        self.sequencer = Sequencer(self.settings.session.sequencer)
        self.sequencer.add_state_callback(self.board.set_sequence)
        self.sequencer.add_state_callback(self.osc_sound.set_sequencer_state)
        self.video_recorder = VideoRecorder(self.settings.session.video, data_path=DATA_PATH)

        # CAMERA
        self.cameras: list[Camera | Simulator] = []
        self.player: Optional[Player] = None
        if self.settings.camera.sim_enabled:
            self.player = Player(self.settings.camera.simulator, data_path=DATA_PATH)
            for i in range(num_cameras):
                self.cameras.append(Simulator(self.player, self.settings.camera.cameras[i], self.settings.camera.simulator))
        else:
            for i in range(num_cameras):
                self.cameras.append(Camera(self.settings.camera.cameras[i]))
        self.frame_sync_bang = Sync(self.settings.camera.frame_sync, False, 'frame_sync')
        self.tracker = PanoramicTracker(self.settings.camera.tracker, num_players, num_cameras)
        self.tracklet_sync_bang = Sync(self.settings.camera.tracklet_sync, False, 'tracklet_sync')
        self.source_uploader = source.Uploader()
        self.crop_extractor = crop.Extractor(ps.image_crop)

        for camera in self.cameras:
            camera.add_sync_callback(self.video_recorder.submit_synced_frames)
            camera.add_frame_callback(self.source_uploader.set_image)
            camera.add_frame_callback(self.frame_sync_bang.submit_frame)
            camera.add_tracker_callback(self.tracker.submit_cam_tracklets)
            camera.add_tracker_callback(self.board.set_depth_tracklets)
            camera.add_tracker_callback(self.tracklet_sync_bang.submit_frame)

        # DETECTION
        features.configure_features(num_players)

        self.poses_from_tracklets = PosesFromTracklets(num_players)

        self.pose_predictor = pose.Predictor(ps.pose)
        self.segmentation_predictor  = segmentation.Predictor(ps.segmentation)

        self.tracker.add_tracklet_callback(self.poses_from_tracklets.set_tracklets)
        self.tracker.add_tracklet_callback(self.board.set_tracklets)
        self.tracklet_sync_bang.add_sync_callback(self.tracker.notify_update)
        self.frame_sync_bang.add_sync_callback(self.poses_from_tracklets.process)

        self.crop_extractor.add_image_callback(self.pose_predictor.process)
        self.crop_extractor.add_image_callback(lambda _f, gpu: self.board.set_crop_images(gpu))
        self.crop_extractor.add_image_callback(self.segmentation_predictor.process)
        self.segmentation_predictor.add_segmentation_image_callback(lambda _f, masks: self.board.set_segmentation_images(masks))

        self.poses_from_tracklets.add_frames_callback(self._process_poses)

        # STAGE WINDOW TRACKERS & BROADCASTS
        # The LERP tracker also windows the app-local playhead features (the only stage where
        # they're stamped) so the data layers can graph them; other stages use the built-ins.
        lerp_features = features.SCALAR_FEATURES + [PlayheadOffset, PlayheadStability, GhostedFeature]
        self.window_trackers: dict[Stage, window.WindowTracker] = {}
        self.stages: dict[Stage, Broadcast] = {}
        for stage in Stage:
            wt_features = lerp_features if stage == Stage.LERP else None
            wt = window.WindowTracker(num_players, getattr(ps, f'window_{stage.name.lower()}'), features=wt_features)
            wt.add_windows_callback(partial(self.board.set_windows, stage))
            self.window_trackers[stage] = wt
            self.stages[stage] = Broadcast([
                partial(self.board.set_frames, stage),
                wt.process,
            ])

        # WS PIPELINE — light output
        ws_input: Stage = Stage(int(ps.ws_input_stage))
        self.light_renderer = LightRender(self.settings.light, distortion=self.settings.camera.tracker.distortion, board=self.board, pose_stage=int(ws_input))
        self.osc_light    = OscLight(self.settings.inout.osc_light)
        self.osc_receiver = OscReceiver(self.settings.inout.osc_receiver)
        self.udp_receiver = UdpReceiver(self.settings.inout.udp_receiver)
        self.osc_receiver.bind("/WS/sensor/fall", self.light_renderer.notify_fall)
        self.udp_receiver.bind("/WS/sensor/fall", self.light_renderer.notify_fall)
        for camera in self.cameras:
            camera.add_frame_callback(self._store_video_frame)
        self.light_renderer.add_render_callback(self.osc_light.send_message)
        self.light_renderer.add_render_callback(self.osc_sound.set_composition)

        # POSE STAGE RAW
        self.pose_predictor.add_frames_callback(self.stages[Stage.RAW])

        # POSE STAGE CLEAN
        self.filters_clean = trackers.FilterTracker({
            i: trackers.FilterPipeline([
                nodes.PointDualConfFilter(ps.point.confidence),
                nodes.PointStickyFiller(ps.point.sticky),
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

        # Pose similarity (enabled by default); movement correlation (disabled by default)
        self.window_similator  = analytics.WindowSimilarity(ps.similarity.window_similarity)
        self.window_correlator = analytics.WindowCorrelation(ps.similarity.window_correlation)

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
                nodes.PointChaseInterpolator(ps.point.interpolator),
                nodes.AngleChaseInterpolator(ps.angle.interpolator),
                nodes.AngleVelChaseInterpolator(ps.velocity.interpolator),
                nodes.SimilarityChaseInterpolator(ps.similarity.interpolator),
            ])
            for i in range(num_players)
        })
        self.filters_lerp = trackers.FilterTracker({
            i: trackers.FilterPipeline([
                nodes.DistanceExtractor(ps.distance_extractor),
                nodes.AngleSymExtractor(),
                nodes.MotionTimeExtractor(),
                nodes.AgeExtractor(),
                nodes.AngleVelStickyFiller(ps.velocity.sticky),
                nodes.AngleVelEuroSmoother(ps.velocity.smoother),
                nodes.AngleMotionExtractor(ps.motion.extractor),
                nodes.AngleMotionMovingAverageSmoother(ps.motion.moving_average),
                PlayheadOffsetExtractor(self.board.get_playhead),
                PlayheadStabilityExtractor(ps.playhead_stability),
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
        # Ghoster sits between LERP and the fan-out: tags GhostedFeature on live frames, spawns ghosts,
        # publishes the ghost snapshot to the board, and feeds (muted live + ghosts) to OSC sound.
        self.gate_lerp.add_frames_callback(self.ghoster.process)
        self.ghoster.add_frames_callback(self.stages[Stage.LERP])
        self.ghoster.add_ghosts_callback(self.board.set_ghosts)
        self.ghoster.add_sound_callback(partial(self.osc_sound.set_frames, int(Stage.LERP)))

        # RENDER
        self.render = WindowRender(self.board, self.settings.render)
        self.settings.render.window.bind(WindowSettings.avg_fps, self._on_render_fps)
        self.light_renderer.add_update_callback(self.sequencer.update)
        self.light_renderer.add_update_callback(self.interpolators_lerp.update)
        self.render.add_exit_callback(self.stop)

    def start(self) -> None:
        self.settings_server.start()

        for camera in self.cameras:
            camera.start()

        self.tracker.start()
        self.pose_predictor.start()
        self.segmentation_predictor.start()
        self.window_similator.start()
        self.window_correlator.start()
        self.light_renderer.start()
        self.osc_light.start()
        self.osc_receiver.start()
        self.udp_receiver.start()

        self.osc_sound.start()

        if self.player:
            self.player.start()
        self.video_recorder.start()

        self.is_running = True
        self.render.start()

    def _store_video_frame(self, cam_id: int, frame_type: FrameType, frame: np.ndarray) -> None:
        """Camera frame callback — store raw VIDEO frames on the board for the light renderer."""
        if frame_type == FrameType.VIDEO:
            self.board.set_video_image(cam_id, frame)

    def _process_poses(self, poses: FrameDict) -> None:
        images, prev_images = self.source_uploader.snapshot()
        self.board.set_camera_images(images)
        self.crop_extractor.process(poses, images, prev_images)

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
        self.osc_sound.stop()
        self.ghoster.stop()
        self.osc_light.stop()
        self.osc_receiver.stop()
        self.udp_receiver.stop()
        self.light_renderer.stop()

        self.pose_predictor.stop()
        self.segmentation_predictor.stop()
        self.window_similator.stop()
        self.window_correlator.stop()

        for camera in self.cameras:
            camera.join(timeout=10)

        self.is_finished = True

    def _on_render_fps(self, fps: int) -> None:
        if fps > 0:
            self.settings.render_fps = float(fps)
