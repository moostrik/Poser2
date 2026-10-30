"""White Space — 3-camera panoramic installation with circular LED light output."""

import math
from pathlib import Path
from functools import partial

import glfw
import numpy as np

from modules.camera import CameraSource, CameraFrame, FrameType, Player, Sync, Recorder as VideoRecorder, CameraCheck, replay
from modules.camera.oak import Camera, Simulator
from modules.settings import presets
from modules.panel import PanelServer
from modules.inout import OscReceiver
from modules.tracker import PanoramicTracker, PosesFromTracklets
from modules.pose import features
from modules.pose.pipeline import Domain, PosePipeline
from modules.inference import detection, PoseInference
from modules.session import Session

from .board import RenderBoard
from .pose import Dummy, dummy_id
from .pose.pipeline import pipeline_spec      # not through the package: it reads the settings, which read the package
from .light import Conductor, Tick
from .inout import OscLightSender, OscSoundSender, UdpLightReceiver
from .render import WhiteSpaceRender
from .settings import Settings, Stage
from .statemachine import StateMachine
from .tracker_journal import TrackerJournal

APP_NAME = 'white_space'
DATA_PATH = 'apps/white_space/data'

import logging
logger = logging.getLogger(__name__)


class WhiteSpaceMain:
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
        # The crop is the pose model's input, so it is cut at that size: one antialiased resample
        # from the camera frame, rather than a larger crop the runner scales down again. Derived
        # from the model so a change of pose resolution cannot silently reintroduce the second one.
        crop_settings = self.settings.pose.image_crop
        crop_settings.output_width = self.settings.pose.pose.width
        crop_settings.output_height = self.settings.pose.pose.height
        self.settings.initialize()
        self.panel_server = PanelServer(self.settings, self.settings.panel, on_exit=self.stop)

        max_players: int = self.settings.max_players
        num_cams: int = self.settings.camera.num_cams
        logging.info("Settings loaded: %s players, %s cameras, simulation=%s", max_players, num_cams, simulation)
        ps = self.settings.pose

        # BLACKBOARD
        self.board = RenderBoard()

        # RECORDING (independent of the show — works stand-alone and during session mode)
        self.session = Session(self.settings.session.control)
        self.osc_sound_sender = OscSoundSender(self.settings.inout.osc_sound_sender)
        # The dummy has its own id above the live players: a source of the LERP stage, polled on the light
        # tick, so it is a pose like any other from there.
        self.dummy = Dummy(self.settings.pose_instrument.dummy, ps.angle_extractor, ps.angle_calibrator, dummy_id(max_players),
                           Path(DATA_PATH) / "poses.json")
        self.video_recorder = VideoRecorder(self.settings.session.video, data_path=DATA_PATH)

        # CAMERA
        self.cameras: list[CameraSource] = []
        self.player: Player | None = None
        self.person_detector: detection.PersonDetector | None = None      # the replay backend's, shared by its cameras
        if self.settings.camera.sim_enabled:
            # The rig is uniform: mono cameras decode their clips straight to gray
            gray = not self.settings.camera.cameras[0].color
            self.player = Player(self.settings.camera.simulator, data_path=DATA_PATH, gray=gray)
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
        self.tracker = PanoramicTracker(self.settings.tracker, max_players, num_cams)
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

        self.tracker.add_tracklet_callback(self.poses_from_tracklets.set_tracklets)
        self.tracker.add_tracklet_callback(self.board.set_tracklets)
        # The unfused view, for the calibration display only — two observations of one person on
        # a seam, which the line above reduces to one.
        self.tracker.add_observation_callback(self.board.set_observations)
        self.tracklet_sync_bang.add_sync_callback(self.tracker.update)
        self.frame_sync_bang.add_sync_callback(self.poses_from_tracklets.process)

        # DIAGNOSIS — a subscribe-only journal of the tracker's inputs, observations and emitted
        # primaries (JSONL under data/diagnosis), for dissecting identity incidents offline.
        # Always wired; `session.journal.enabled` (panel, preset or `launcher.py -journal`) toggles
        # capture live, one folder per capture, and the taps return on a flag check while off.
        self.tracker_journal = TrackerJournal(self.settings.session.journal, data_path=DATA_PATH,
                                              meta_provider=self._journal_meta,
                                              track_provider=self.settings.tracker.to_dict)
        for camera in self.cameras:
            camera.add_tracklet_callback(self.tracker_journal.submit_cam_tracklets)
        self.tracker.add_observation_callback(self.tracker_journal.submit_observations)
        self.tracker.add_tracklet_callback(self.tracker_journal.submit_tracklets)

        self.pose_inference.add_camera_images_callback(self.board.set_camera_images)
        self.pose_inference.add_crop_callback(lambda _f, gpu: self.board.set_crop_images(gpu))

        # POSE PIPELINE — the stages are declared in pose/pipeline.py. The camera's frame bang drives the frame
        # domain through the tracklet poses; the light tick drives the output domain (LERP, where the dummy joins).
        # Every stage's frames and group record go to the board, which also reads each stage's window; the sound
        # reads LERP, and so do the instrument and the state machine (the group sync on the LERP record).
        self.pipeline = PosePipeline(pipeline_spec(ps, self.settings.light, self.poses_from_tracklets, self.pose_inference,
                                                   self.dummy, self._column_to_azimuth, self.board.get_playhead_phase,
                                                   self.board.get_playhead_bars))
        for stage in Stage:
            self.pipeline.add_frames_callback(stage, partial(self.board.set_frames, stage))
            self.pipeline.add_group_callback(stage, partial(self.board.set_group, stage))
            self.board.set_window_source(stage, self.pipeline.window(stage))
        self.pipeline.add_frames_callback(Stage.LERP, partial(self.osc_sound_sender.submit_frames, int(Stage.LERP)))

        # WS PIPELINE — light output
        # The show reads LERP poses: the stable eye azimuth, and the only stage with PlayheadOffset.
        self.conductor = Conductor(self.settings.light, self.settings.pose_instrument, board=self.board, pose_stage=int(Stage.LERP))
        # One receiver per domain, matching each source's actual transport: the fixture
        # firmware sends the fall as a plain UDP text packet (not OSC) to the light
        # receiver's port; Max sends /WS/idle/blue/left and /right as real OSC (the UDP
        # receiver could never decode their float args). Don't cross-bind them.
        self.osc_light_sender   = OscLightSender(self.settings.inout.osc_light_sender, self.settings.light.motor)
        self.udp_light_receiver = UdpLightReceiver(self.settings.inout.udp_light_receiver)
        self.osc_sound_receiver = OscReceiver(self.settings.inout.osc_sound_receiver)
        self.udp_light_receiver.bind("/WS/sensor/fall", self.conductor.notify_fall)
        # Sound levels from Max (one 0..1 float each) → board → the beam_blue_sound layer.
        self.osc_sound_receiver.bind("/WS/idle/blue/left",  self._on_sound_level_left)
        self.osc_sound_receiver.bind("/WS/idle/blue/right", self._on_sound_level_right)
        for camera in self.cameras:
            camera.add_frame_callback(self._store_video_frame)
        self.conductor.add_light_frame_callback(self.osc_light_sender.submit_light_frame)
        self.conductor.add_light_frame_callback(self.osc_sound_sender.set_light_frame)

        # STATE MACHINE — the show's single decision maker; commands the Conductor through
        # its three channels (look, layer resets, motor) and emits state to board + OSC.
        self.state_machine = StateMachine(
            self.settings.states, self.settings.light, board=self.board,
            set_mix=self.conductor.set_mix,
            reset_layers=self.conductor.reset_layers,
            set_motor=self.conductor.set_motor_mode,
            pose_stage=int(Stage.LERP),
        )
        self.state_machine.add_state_callback(self.board.set_sequencer_state)
        self.state_machine.add_state_callback(self.osc_sound_sender.set_sequencer_state)

        # CAMERA CHECK — compares each camera's own IMU reading and frame rate against the preset,
        # and summarises them as pinned indicators. A setup aid; nothing downstream reads it.
        self.camera_check = CameraCheck(self.settings.camera.cameras, self.settings.camera.camera_check)

        # RENDER
        self.render = WhiteSpaceRender(self.board, self.settings.render, self.settings.tracker,
                                       self.settings.camera.cameras, self.settings.camera.camera_check, ps.angle_extractor)
        self.render.add_update_callback(self.camera_check.update)
        # In order: LERP first, which stamps this tick's hit on the poses and writes the streak and the group
        # sync to the board; then the state machine reads them; then the layers draw on the same.
        self.conductor.add_update_callback(self._update_poses)
        self.conductor.add_update_callback(self.state_machine.update)
        self.render.add_exit_callback(self.stop)
        self.render.window_manager.add_keyboard_callback(self._on_key)

    def start(self) -> None:
        self.panel_server.start()

        if self.person_detector is not None:
            self.person_detector.start()
        for camera in self.cameras:
            camera.start()

        self.tracker.start()
        self.pose_inference.start()
        self.conductor.start()
        self.osc_light_sender.start()
        self.osc_sound_receiver.start()
        self.udp_light_receiver.start()

        self.osc_sound_sender.start()

        if self.player:
            self.player.start()
        self.video_recorder.start()
        self.tracker_journal.start()

        self.is_running = True
        self.render.start()

    def _on_sound_level_left(self, level: float = 0.0, *_rest: object) -> None:
        """OSC callback for /WS/idle/blue/left — store the Max soundscape's left level."""
        self.board.set_sound_level_left(float(level))

    def _on_sound_level_right(self, level: float = 0.0, *_rest: object) -> None:
        """OSC callback for /WS/idle/blue/right — store the Max soundscape's right level."""
        self.board.set_sound_level_right(float(level))

    def _store_video_frame(self, frame: CameraFrame) -> None:
        """Camera frame callback — store raw VIDEO frames on the board for the light renderer."""
        if frame.kind == FrameType.VIDEO:
            self.board.set_video_image(frame.cam_id, frame.image)

    def _column_to_azimuth(self, cam_id: int, x: float) -> float:
        """A camera column's world azimuth in radians, through the tracker's own geometry."""
        return math.radians(self.tracker.column_to_azimuth(cam_id, x))

    def _on_key(self, key: int, action: int, _mods: int) -> None:
        """Render-window hotkeys: M marks an incident in the tracker journal."""
        if key == glfw.KEY_M and action == glfw.PRESS:
            self.tracker_journal.mark()

    def _journal_meta(self) -> dict:
        """The journal's meta line: the numbers an offline analysis recomputes the gates from."""
        sim = self.settings.camera.simulator
        return {
            'app': APP_NAME,
            'num_cams': self.settings.camera.num_cams,
            'max_players': self.settings.max_players,
            'fps': self.settings.camera.fps,
            'track': self.settings.tracker.to_dict(),
            'source': {'folder': sim.folder, 'start': sim.in_out[0], 'end': sim.in_out[1]}
                      if self.player else None,
        }

    def _update_poses(self, tick: Tick) -> None:
        """The light tick drives the pipeline's output domain (LERP) at the time the tick was paced for, so the
        LERP frames carry a regular stamp whatever the tick's lateness."""
        self.pipeline.update(Domain.OUTPUT, tick.deadline)

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

        self.tracker.stop()
        self.tracker_journal.stop()
        self.osc_sound_sender.stop()
        self.state_machine.stop()
        self.osc_light_sender.stop()
        self.osc_sound_receiver.stop()
        self.udp_light_receiver.stop()
        self.conductor.stop()

        self.pose_inference.stop()
        if self.person_detector is not None:
            self.person_detector.stop()

        for camera in self.cameras:
            camera.join(timeout=10)

        self.is_finished = True
