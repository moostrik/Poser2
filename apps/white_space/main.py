"""White Space — 3-camera panoramic installation with circular LED light output."""

import math
from typing import Optional
from functools import partial

import numpy as np

from modules.utils import Broadcast
from modules.oak import Camera, Simulator, Player, Sync, Recorder as VideoRecorder, FrameType, CameraCheck, delivered_height
from modules.settings import presets, NiceServer
from modules.inout import OscReceiver
from modules.tracker import PanoramicTracker, PosesFromTracklets
from modules.pose import nodes, trackers, features, window, analytics, FrameDict
from modules.inference import source, crop, pose
from modules.session import Session
from modules.gl import WindowSettings

from .board import Board
from .pose import GhostFeature, PlayheadOffset, PlayheadOffsetExtractor, Ghoster
from .light import Conductor
from .inout import OscLightSender, OscSoundSender, UdpLightReceiver
from .render import Render as WindowRender
from .settings import Settings, Stage
from .statemachine import StateMachine

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
        # The delivered frame's height follows the tilt unless the preset pins it. The warp's
        # rows are tangents of elevation, so the sensor's full reach needs more rows the further
        # the camera is aimed up (848 at P720 and tilt 0, 960 at tilt 15, 1152 at P800 and tilt
        # 16). Derived once, here, before anything sizes itself by it; a non-zero preset value
        # is an explicit override. See CALIBRATION.md, "The camera frame".
        if self.settings.frame_height == 0:
            self.settings.frame_height = delivered_height(
                self.settings.camera.color, self.settings.resolution, self.settings.fov, self.settings.tilt,
                self.settings.lens_fov, (self.settings.lens_centre_x, self.settings.lens_centre_y))
            logging.info("frame_height derived: %d rows for %s at tilt %.1f", self.settings.frame_height,
                         self.settings.resolution.name, self.settings.tilt)
        # The crop is the pose model's input, so it is cut at that size: one antialiased resample
        # from the camera frame, rather than a larger crop the runner scales down again. Derived
        # from the model so a change of pose resolution cannot silently reintroduce the second one.
        crop_settings = self.settings.pose.image_crop
        crop_settings.output_width = self.settings.pose.pose.width
        crop_settings.output_height = self.settings.pose.pose.height
        self.settings.initialize()
        self.settings_server = NiceServer(self.settings, self.settings.server, on_exit=self.stop)

        num_players: int = self.settings.num_players
        num_cameras: int = self.settings.camera.num_cameras
        logging.info("Settings loaded: %s players, %s cameras, simulation=%s", num_players, num_cameras, simulation)
        ps = self.settings.pose

        # BLACKBOARD
        self.board = Board()

        # RECORDING (independent of the show — works stand-alone and during session mode)
        self.session = Session(self.settings.record.core)
        self.osc_sound_sender = OscSoundSender(self.settings.inout.osc_sound_sender)
        self.ghoster = Ghoster(self.settings.pose.ghoster, playhead=self.board.get_playhead)   # live/pool counts shared from root
        self.video_recorder = VideoRecorder(self.settings.record.video, data_path=DATA_PATH)

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
        self.tracker = PanoramicTracker(self.settings.track, num_players, num_cameras)
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

        self.poses_from_tracklets = PosesFromTracklets(ps.tracklets, num_players)

        self.pose_predictor = pose.Predictor(ps.pose)

        self.tracker.add_tracklet_callback(self.poses_from_tracklets.set_tracklets)
        self.tracker.add_tracklet_callback(self.board.set_tracklets)
        # The unfused view, for the calibration display only — two observations of one person on
        # a seam, which the line above reduces to one.
        self.tracker.add_observation_callback(self.board.set_observations)
        self.tracklet_sync_bang.add_sync_callback(self.tracker.notify_update)
        self.frame_sync_bang.add_sync_callback(self.poses_from_tracklets.process)

        self.crop_extractor.add_image_callback(self.pose_predictor.process)
        self.crop_extractor.add_image_callback(lambda _f, gpu: self.board.set_crop_images(gpu))

        self.poses_from_tracklets.add_frames_callback(self._process_poses)

        # STAGE WINDOW TRACKERS & BROADCASTS
        # The LERP tracker also windows the app-local playhead features (the only stage where
        # they're stamped) so the data layers can graph them; other stages use the built-ins.
        lerp_features = features.SCALAR_FEATURES + [PlayheadOffset, GhostFeature]
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
        # The show reads LERP poses: the stable eye azimuth, and the only stage with PlayheadOffset.
        self.conductor = Conductor(self.settings.light, board=self.board, pose_stage=int(Stage.LERP))
        # One receiver per domain, matching each source's actual transport: the fixture
        # firmware sends the fall as a plain UDP text packet (not OSC) to the light
        # receiver's port; Max sends /WS/sound/level as real OSC (the UDP receiver could
        # never decode its float args). Don't cross-bind them.
        self.osc_light_sender   = OscLightSender(self.settings.inout.osc_light_sender)
        self.udp_light_receiver = UdpLightReceiver(self.settings.inout.udp_light_receiver)
        self.osc_sound_receiver = OscReceiver(self.settings.inout.osc_sound_receiver)
        self.udp_light_receiver.bind("/WS/sensor/fall", self.conductor.notify_fall)
        # Sound levels from Max (left, right 0..1) → board → the beam_blue_sound layer.
        self.osc_sound_receiver.bind("/WS/sound/level", self._on_sound_level)
        for camera in self.cameras:
            camera.add_frame_callback(self._store_video_frame)
        self.conductor.add_render_callback(self.osc_light_sender.send_message)
        self.conductor.add_render_callback(self.osc_sound_sender.set_composition)

        # STATE MACHINE — the show's single decision maker; commands the Conductor through
        # its three channels (look, layer resets, motor) and emits state to board + OSC.
        self.state_machine = StateMachine(
            self.settings.states, self.settings.light, board=self.board,
            set_mix=self.conductor.set_mix,
            reset_layers=self.conductor.reset_layers,
            set_motor=self.conductor.set_motor_mode,
            pose_stage=int(Stage.LERP),
        )
        self.state_machine.add_state_callback(self.board.set_sequence)
        self.state_machine.add_state_callback(self.osc_sound_sender.set_sequencer_state)

        # POSE STAGE RAW
        self.pose_predictor.add_frames_callback(self.stages[Stage.RAW])

        # POSE STAGE CLEAN
        self.filters_clean = trackers.FilterTracker({
            i: trackers.FilterPipeline([
                nodes.PointDualConfFilter(ps.point.confidence),
                nodes.PointStickyFiller(ps.point.sticky),
                # Here, not later: the keypoints are still normalised in the crop they came from, so
                # the box's jitter cancels out of the eye azimuth. A held eye moves with the box.
                nodes.AzimuthExtractor(self._column_to_azimuth),
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
                nodes.AzimuthEuroSmoother(ps.azimuth.smoother),
                nodes.AngleMotionExtractor(ps.motion.extractor),
                nodes.AngleMotionMovingAverageSmoother(ps.motion.moving_average),
                nodes.AngleSymExtractor(),
                nodes.LegDeviationExtractor(ps.leg_deviation_extractor),
                nodes.TorsoTiltExtractor(ps.torso_tilt_extractor),
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
        self.window_similator.add_similarity_callback(self.state_machine.set_similarity)

        self.window_trackers[Stage.SMOOTH].add_windows_callback(self.window_correlator.submit)
        self.window_correlator.add_similarity_callback(self.similarity_applicator.set)
        self.window_correlator.add_similarity_callback(self.leader_applicator.set)
        self.window_correlator.add_similarity_callback(self.state_machine.set_correlation)

        # POSE STAGE PREDICT
        self.filters_predict = trackers.FilterTracker({
            i: trackers.FilterPipeline([
                nodes.PointPredictor(ps.point.prediction),
                nodes.AnglePredictor(ps.angle.prediction),
                nodes.AngleVelPredictor(ps.velocity.prediction),
                nodes.AzimuthPredictor(ps.azimuth.prediction),
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
                nodes.AzimuthChaseInterpolator(ps.azimuth.interpolator),
            ])
            for i in range(num_players)
        })
        self.filters_lerp = trackers.FilterTracker({
            i: trackers.FilterPipeline([
                nodes.AngleSymExtractor(),
                nodes.LegDeviationExtractor(ps.leg_deviation_extractor),
                nodes.TorsoTiltExtractor(ps.torso_tilt_extractor),
                nodes.MotionTimeExtractor(),
                nodes.AgeExtractor(),
                nodes.AngleVelStickyFiller(ps.velocity.sticky),
                nodes.AngleVelEuroSmoother(ps.velocity.smoother),
                nodes.AngleMotionExtractor(ps.motion.extractor),
                nodes.AngleMotionMovingAverageSmoother(ps.motion.moving_average),
                PlayheadOffsetExtractor(self.board.get_playhead),
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
        # Ghoster sits between LERP and the fan-out: records held poses, commits/refreshes ghosts,
        # publishes the ghost snapshot to the board, and feeds (muted live + ghosts) to OSC sound.
        self.gate_lerp.add_frames_callback(self.ghoster.process)
        self.ghoster.add_frames_callback(self.stages[Stage.LERP])
        self.ghoster.add_ghosts_callback(self.board.set_ghosts)
        self.ghoster.add_sound_callback(partial(self.osc_sound_sender.set_frames, int(Stage.LERP)))

        # CAMERA CHECK — compares each camera's own IMU reading and frame rate against the preset,
        # and summarises them as pinned indicators. A setup aid; nothing downstream reads it.
        self.camera_check = CameraCheck(self.settings.camera.cameras, self.settings.camera.camera_check)

        # RENDER
        self.render = WindowRender(self.board, self.settings.render, self.settings.track,
                                   self.settings.camera.cameras, self.settings.camera.camera_check)
        self.settings.render.window.bind(WindowSettings.avg_fps, self._on_render_fps)
        self.render.add_update_callback(self.camera_check.update)
        # LERP first: the state machine's hit reads this tick's PlayheadOffset, the same the flash draws on.
        self.conductor.add_update_callback(self.interpolators_lerp.update)
        self.conductor.add_update_callback(self.state_machine.update)
        self.render.add_exit_callback(self.stop)

    def start(self) -> None:
        self.settings_server.start()

        for camera in self.cameras:
            camera.start()

        self.tracker.start()
        self.pose_predictor.start()
        self.window_similator.start()
        self.window_correlator.start()
        self.conductor.start()
        self.osc_light_sender.start()
        self.osc_sound_receiver.start()
        self.udp_light_receiver.start()

        self.osc_sound_sender.start()

        if self.player:
            self.player.start()
        self.video_recorder.start()

        self.is_running = True
        self.render.start()

    def _on_sound_level(self, left: float = 0.0, right: float = 0.0, *_rest: object) -> None:
        """OSC/UDP callback for /WS/sound/level — store the Max soundscape levels."""
        self.board.set_sound_levels(float(left), float(right))

    def _store_video_frame(self, cam_id: int, frame_type: FrameType, frame: np.ndarray) -> None:
        """Camera frame callback — store raw VIDEO frames on the board for the light renderer."""
        if frame_type == FrameType.VIDEO:
            self.board.set_video_image(cam_id, frame)

    def _column_to_azimuth(self, cam_id: int, x: float) -> float:
        """A camera column's world azimuth in radians, through the tracker's own geometry."""
        return math.radians(self.tracker.column_to_azimuth(cam_id, x))

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
        self.osc_sound_sender.stop()
        self.state_machine.stop()
        self.ghoster.stop()
        self.osc_light_sender.stop()
        self.osc_sound_receiver.stop()
        self.udp_light_receiver.stop()
        self.conductor.stop()

        self.pose_predictor.stop()
        self.window_similator.stop()
        self.window_correlator.stop()

        for camera in self.cameras:
            camera.join(timeout=10)

        self.is_finished = True

    def _on_render_fps(self, fps: int) -> None:
        if fps > 0:
            self.settings.render_fps = float(fps)
