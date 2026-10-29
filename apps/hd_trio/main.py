"""HD Trio — 3-camera interactive installation with fluid rendering."""

import logging
from functools import partial

from modules.camera import CameraSource, CameraCheck, Player, Sync, Recorder as VideoRecorder, replay
from modules.camera.oak import Camera, Simulator
from modules.settings import presets
from modules.panel import PanelServer
from modules.inout import OscSound, ArtNetBars, OscReceiver
from modules.tracker import OnePerCamTracker, PosesFromTracklets
from modules.pose import nodes, features, analytics, FrameDict, Recorder as PoseRecorder, RecorderSettings as PoseRecorderSettings
from modules.pose.pipeline import Domain, PosePipeline
from modules.inference import segmentation, detection, PoseInference
from modules.session import Session, SessionSettings, SequencerState
from modules.gl import WindowSettings

from .render_board import RenderBoard
from .pose.pipeline import pipeline_spec
from .sequencer import Sequencer
from .settings import Settings, Stage, ShowStage, IntroSequenceSettings
from .render import HDTrioRender

logger = logging.getLogger(__name__)

APP_NAME = 'hd_trio'
DATA_PATH = 'apps/hd_trio/data'


class HDTrioMain:
    def __init__(self, simulation: bool = False, interface: bool = True) -> None:

        self.is_running: bool = False
        self.is_finished: bool = False

        # SETTINGS
        presets.set_app(APP_NAME)
        self.settings = Settings()
        preset_file = presets.startup_path()
        if not presets.load(self.settings, preset_file):
            raise FileNotFoundError(f"No preset found for '{APP_NAME}' at {preset_file}")
        self.settings.camera.sim_enabled = simulation
        if not interface:
            self.settings.panel.browser = False  # launcher override; the preset cannot turn it back on
        # Never resume armed: a preset saved mid-take would otherwise record the
        # first show after startup.
        self.settings.render.intro_sequence.arm_record = False
        self.settings.initialize()
        self.panel_server = PanelServer(self.settings, self.settings.panel, on_exit=self.stop)

        max_players: int = self.settings.max_players
        num_cams: int = self.settings.num_cams
        ps = self.settings.pose

        # BLACKBOARD
        self.board = RenderBoard()

        # SESSION
        self.session = Session(self.settings.session.control)
        self.session_osc = OscReceiver(self.settings.session.osc)
        self.session_osc.bind('/start/recording', self._on_osc_start_recording)
        self.session_osc.bind('/stop/recording',  self._on_osc_stop_recording)
        self.session_osc.bind('/group/id',        self._on_osc_group_id)
        self.osc_sound = OscSound(self.settings.inout.osc_sound)
        self.sequencer = Sequencer(self.settings.session.sequencer)
        self.sequencer.add_state_callback(self.board.set_sequencer_state)
        self.sequencer.add_state_callback(self.osc_sound.set_sequencer_state)
        self.video_recorder = VideoRecorder(self.settings.session.video, data_path=DATA_PATH)
        self.pose_recorder = PoseRecorder(self.settings.session.pose, data_path=DATA_PATH)

        # Authoring recorder for the white example figure. Deliberately not part of
        # the session share group — its start/stop must not reach the sequencer.
        self.intro_recorder = PoseRecorder(self.settings.render.intro_sequence.recorder, data_path=DATA_PATH)
        self._prev_in_record_window: bool = False
        self.sequencer.add_state_callback(self._on_show_state)
        self.settings.render.intro_sequence.bind(IntroSequenceSettings.arm_record, self._on_arm_record)
        self.artnet_controllers: list[ArtNetBars] = []
        for i in range(max_players):
            self.artnet_controllers.append(ArtNetBars(self.settings.inout.artnets[i]))

        # CAMERA
        self.cameras: list[CameraSource] = []
        self.player: Player | None = None
        self.person_detector: detection.PersonDetector | None = None      # the replay backend's, shared by its cameras
        if self.settings.camera.sim_enabled:
            self.player = Player(self.settings.camera.simulator, data_path=DATA_PATH)
            if self.settings.camera.simulator.replay:
                # the recordings on the host, without a device: the people found and followed here
                if self.settings.camera.yolo:
                    self.person_detector = detection.PersonDetector(self.settings.camera.person_detector)
                for i in range(num_cams):
                    self.cameras.append(replay.Camera(self.player, self.settings.camera.cameras[i], self.person_detector, replay.Tracker(), i))
            else:
                for i in range(num_cams):
                    self.cameras.append(Simulator(self.player, self.settings.camera.cameras[i], self.settings.camera.simulator))
        else:
            for i in range(num_cams):
                self.cameras.append(Camera(self.settings.camera.cameras[i]))
        self.frame_sync_bang = Sync(self.settings.camera.frame_sync, False, 'frame_sync')
        self.tracker = OnePerCamTracker(self.settings.camera.tracker, max_players)
        self.tracklet_sync_bang = Sync(self.settings.camera.tracklet_sync, False, 'tracklet_sync')
        self.pose_inference = PoseInference(ps.image_crop, ps.pose)      # the crop and the pose model, the frame domain's async step

        for camera in self.cameras:
            camera.add_frame_callback(self.video_recorder.submit_frame)
            camera.add_frame_callback(self.pose_inference.set_camera_frame)
            camera.add_frame_callback(self.frame_sync_bang.submit_frame)
            camera.add_tracklet_callback(self.tracker.submit_cam_tracklets)
            camera.add_tracklet_callback(self.board.set_camera_tracklets)
            camera.add_tracklet_callback(self.tracklet_sync_bang.submit_tracklets)

        # DETECTION
        features.configure_features(max_players)

        self.poses_from_tracklets = PosesFromTracklets(ps.tracklets, max_players)
        self.segmentation_predictor  = segmentation.Predictor(ps.segmentation)

        self.tracker.add_tracklet_callback(self.poses_from_tracklets.set_tracklets)
        self.tracklet_sync_bang.add_sync_callback(self.tracker.update)
        self.frame_sync_bang.add_sync_callback(self.poses_from_tracklets.process)

        # The segmentation and the board read the same crops the pose model gets.
        self.pose_inference.add_camera_images_callback(self.board.set_camera_images)
        self.pose_inference.add_crop_callback(self.segmentation_predictor.submit)
        self.pose_inference.add_crop_callback(lambda _f, crops: self.board.set_crop_images(crops))
        self.segmentation_predictor.add_segmentation_callback(lambda _f, masks: self.board.set_segmentation_images(masks))

        # POSE PIPELINE — the stages are declared in pose/pipeline.py. The camera's frame bang drives the frame
        # domain through the tracklet poses; the render tick drives the output domain (LERP). Every stage's frames
        # go to the board, the recorders and the sound; the board also reads each stage's window.
        self.similarity_hold = nodes.SimilarityResultHold()              # the threaded analytics' way back into SMOOTH
        self.pipeline = PosePipeline(pipeline_spec(ps, self.poses_from_tracklets, self.pose_inference, self.similarity_hold))
        for stage in Stage:
            self.pipeline.add_frames_callback(stage, partial(self.board.set_frames, stage))
            self.pipeline.add_frames_callback(stage, partial(self.pose_recorder.submit_frames, stage))
            self.pipeline.add_frames_callback(stage, partial(self._submit_intro_frames, stage))
            self.pipeline.add_frames_callback(stage, partial(self.osc_sound.submit_frames, stage))
            self.board.set_window_source(stage, self.pipeline.window(stage))

        # The threaded analytics read the SMOOTH window and hand their result to the hold.
        self.window_similarity  = analytics.WindowSimilarity(ps.similarity.window_similarity)
        self.window_correlator = analytics.WindowCorrelation(ps.similarity.window_correlation)
        self.pipeline.window(Stage.SMOOTH).add_windows_callback(self.window_similarity.submit)
        self.window_similarity.add_similarity_callback(self.similarity_hold.set)
        self.pipeline.window(Stage.SMOOTH).add_windows_callback(self.window_correlator.submit)
        self.window_correlator.add_similarity_callback(self.similarity_hold.set)

        # CAMERA CHECK — each camera's IMU reading and frame rate against the preset, as pinned indicators
        self.camera_check = CameraCheck(self.settings.camera.cameras, self.settings.camera.camera_check)

        # RENDER
        self.render = HDTrioRender(self.board, self.settings.render, self.settings.camera.cameras,
                                   self.settings.camera.camera_check, data_path=DATA_PATH)
        self.settings.render.window.bind(WindowSettings.avg_fps, self._on_render_fps)
        self.render.add_update_callback(self.camera_check.update)
        self.render.add_update_callback(self.sequencer.update)
        self.render.add_update_callback(self._update_poses)
        self.render.add_exit_callback(self.stop)

    def start(self) -> None:
        self.panel_server.start()

        if self.person_detector is not None:
            self.person_detector.start()
        for camera in self.cameras:
            camera.start()

        self.tracker.start()
        self.pose_inference.start()
        self.segmentation_predictor.start()
        self.window_similarity.start()
        self.window_correlator.start()

        self.osc_sound.start()
        self.session_osc.start()
        for artnet in self.artnet_controllers:
            artnet.start()

        if self.player:
            self.player.start()
        self.video_recorder.start()

        self.is_running = True
        self.render.start()

    def _update_poses(self, now: float) -> None:
        """The render tick drives the pipeline's output domain (LERP) at the frame's time."""
        self.pipeline.update(Domain.OUTPUT, now)

    def stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False

        self.panel_server.stop()

        self.render.stop()

        if self.player:
            self.player.stop()
        for camera in self.cameras:
            camera.stop()
        self.video_recorder.stop()
        # The pose recorders have no stop() — an in-flight chunk is only written
        # when the stop button fires, so flush it explicitly.
        if self.settings.render.intro_sequence.recorder.recording:
            PoseRecorderSettings.stop.fire(self.settings.render.intro_sequence.recorder)

        self.tracker.stop()
        self.osc_sound.stop()

        for artnet in self.artnet_controllers:
            artnet.stop()

        self.session_osc.stop()

        self.pose_inference.stop()
        if self.person_detector is not None:
            self.person_detector.stop()
        self.segmentation_predictor.stop()
        self.window_similarity.stop()
        self.window_correlator.stop()

        for camera in self.cameras:
            camera.join(timeout=10)

        self.is_finished = True

    def _on_arm_record(self, armed: bool) -> None:
        """Warn at arm time if the configured window can never open."""
        intro = self.settings.render.intro_sequence
        if armed and intro.record_end < intro.record_start:
            logger.warning("intro record window is empty: record_end (%s) is before record_start (%s)",
                           intro.record_end.name, intro.record_start.name)

    def _submit_intro_frames(self, stage: Stage, frames: FrameDict) -> None:
        """Submit only the performer's track, so the intro file holds one figure."""
        track_id: int = self.settings.render.intro_sequence.record_track
        frame = frames.get(track_id)
        self.intro_recorder.submit_frames(stage, {track_id: frame} if frame is not None else {})

    def _on_show_state(self, state: SequencerState) -> None:
        """Start and stop the intro recording on the configured stage window.

        The window is a value range because ShowStage declaration order is show
        order — so leaving it also covers the show being stopped or skipped
        mid-recording, which a 'did we just leave stage X' rule would miss.
        """
        intro = self.settings.render.intro_sequence
        stage = ShowStage(state.stage)
        in_window: bool = intro.record_start <= stage <= intro.record_end
        entering: bool = in_window and not self._prev_in_record_window
        leaving: bool = self._prev_in_record_window and not in_window
        self._prev_in_record_window = in_window

        if entering and intro.arm_record:
            PoseRecorderSettings.start.fire(intro.recorder)
        elif leaving and intro.recorder.recording:
            PoseRecorderSettings.stop.fire(intro.recorder)
            intro.arm_record = False

    def _on_osc_start_recording(self, *_) -> None:
        SessionSettings.start.fire(self.settings.session)

    def _on_osc_stop_recording(self, *_) -> None:
        SessionSettings.stop.fire(self.settings.session)

    def _on_osc_group_id(self, gid: str, *_) -> None:
        self.settings.session.name = gid

    def _on_render_fps(self, fps: int) -> None:
        if fps > 0:
            self.settings.render_fps = float(fps)
