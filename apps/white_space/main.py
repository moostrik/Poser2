"""White Space — 3-camera panoramic installation with circular LED light output."""

import math
import time
from pathlib import Path
from functools import partial

import glfw

from modules.camera import CameraFrame, CameraHealth, CameraSource, CameraTracklets
from modules.camera.oak import Camera, ClipCamera
from modules.recording import Player, Recorder as VideoRecorder, replay
from modules.settings import presets
from modules.panel import PanelServer
from modules.inout import OscReceiver
from modules.tracker import PanoramicTracker, TrackerFrameSource
from modules.pose import FrameDict, features
from modules.pose.pipeline import Domain, PosePipeline
from modules.inference import pose, person, BatchUploader, CameraSync, CropExtractor, frame_poses, has_points
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
        crop_settings = self.settings.pose.pose_crop
        crop_settings.output_width = self.settings.pose.pose.width
        crop_settings.output_height = self.settings.pose.pose.height
        self.settings.initialize()
        self.panel_server = PanelServer(self.settings, self.settings.panel, on_exit=self.stop)

        max_players: int = self.settings.max_players
        num_cams: int = self.settings.camera.num_cams
        logger.info("Settings loaded: %s players, %s cameras, simulation=%s", max_players, num_cams, simulation)
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
        self.video_recorder = VideoRecorder(self.settings.session.video, frame_poses, data_path=DATA_PATH)

        # CAMERA
        self.cameras: list[CameraSource] = []
        self.player: Player | None = None
        self.person_detector: person.PersonDetector | None = None      # the replay backend's, shared by its cameras
        if self.settings.camera.sim_enabled:
            # The rig is uniform: mono cameras decode their clips straight to gray
            gray = not self.settings.camera.cameras[0].color
            self.player = Player(self.settings.camera.simulator, data_path=DATA_PATH, gray=gray)
            if self.settings.camera.simulator.replay:
                # the recordings on the host, without a device: the people found and followed here
                if self.settings.camera.yolo:
                    self.person_detector = person.PersonDetector(self.settings.camera.person_detector)
                for i in range(num_cams):
                    self.cameras.append(replay.Camera(self.player, self.settings.camera.cameras[i], self.person_detector, replay.Tracker(), i))
            else:
                for i in range(num_cams):
                    self.cameras.append(ClipCamera(self.player, self.settings.camera.cameras[i]))
        else:
            for i in range(num_cams):
                self.cameras.append(Camera(self.settings.camera.cameras[i]))
        self.frame_sync: CameraSync[CameraFrame] = CameraSync(self.settings.camera.frame_sync, 'frame_sync')
        self.uploader = BatchUploader()          # a synced set of the cameras' frames onto the GPU as one batch, on its own thread
        self.tracker = PanoramicTracker(self.settings.tracker, max_players, num_cams)
        self.tracklet_sync: CameraSync[CameraTracklets] = CameraSync(self.settings.camera.tracklet_sync, 'tracklet_sync')
        # The frame domain's async step: the crop step in, the pose model out
        self.crops = CropExtractor(ps.pose_crop)
        self.pose_estimator = pose.PoseEstimator(ps.pose)
        self.crops.add_image_callback(self.pose_estimator.submit)

        # The camera's thread only hands its frames and tracklets over: the frames to the recorder and the frames'
        # sync, the tracklets as they come to the tracker, the board and the tracklets' sync (the journal's below)
        for camera in self.cameras:
            camera.add_frame_callback(self.frame_sync.submit)          # the live path first, the recorder after it
            camera.add_frame_callback(self.video_recorder.submit_frame)
            camera.add_tracklet_callback(self.tracker.submit_cam_tracklets)
            camera.add_tracklet_callback(self.board.set_camera_tracklets)
            camera.add_tracklet_callback(self.tracklet_sync.submit)
            camera.add_tracklet_callback(self.video_recorder.submit_tracklets)
        # What each camera's captures wait for from the first: live and through the clip camera, the device sends
        # tracklets (a replay's come or not with the recording); the crop step says per set whether it asked poses
        # (`_on_asked`)
        device_tracklets: bool = not (self.settings.camera.sim_enabled and self.settings.camera.simulator.replay)
        for i in range(num_cams):
            self.video_recorder.expect_parts(i, tracklets=device_tracklets, poses=True)

        # DETECTION
        features.configure_features(max_players)

        self.frame_source = TrackerFrameSource(ps.frame_source, max_players)

        self.tracker.add_tracklet_callback(self.frame_source.set_tracklets)
        self.tracker.add_tracklet_callback(self.board.set_tracklets)
        # The unfused view, for the calibration display only — two observations of one person on
        # a seam, which the line above reduces to one.
        self.tracker.add_observation_callback(self.board.set_observations)
        # The frame source's thread: each set's poses with its images to the pipeline; the crop step in it then
        # tells the poses asked per camera
        self.crops.add_asked_callback(self._on_asked)

        # THE SYNCS' HAND-OFFS, each on its own thread: the frames' set to the uploader, the tracklets' to the tracker
        self.frame_sync.add_set_callback(self.uploader.submit_set)
        self.tracklet_sync.add_set_callback(self._on_tracklet_set)

        # THE UPLOADER'S HAND-OFFS, in order on its thread: each uploaded image to the detector (-sim) and the board,
        # then the set's snapshot to the frame source
        if self.person_detector is not None:
            self.uploader.add_gpu_image_callback(self.person_detector.submit)
        self.uploader.add_finished_gpu_image_callback(self.board.set_camera_image)
        self.uploader.add_snapshot_callback(self.crops.submit_snapshot)
        self.uploader.add_capture_callback(self.frame_source.submit)

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

        self.crops.add_finished_image_callback(lambda _f, gpu: self.board.set_crop_images(gpu))

        # POSE PIPELINE — the stages are declared in pose/pipeline.py. Each uploaded set drives the frame domain
        # through the tracklet poses (the uploader's capture times);
        # the light tick drives the output domain (LERP, where the dummy joins).
        # Every stage's frames and group record go to the board, which also reads each stage's window; the sound
        # reads LERP, and so do the instrument and the state machine (the group sync on the LERP record).
        self.pipeline = PosePipeline(pipeline_spec(ps, self.settings.light, self.frame_source, self.crops, self.pose_estimator,
                                                   self.dummy, self._column_to_azimuth, self.board.get_playhead_phase,
                                                   self.board.get_playhead_bars))
        for stage in Stage:
            self.pipeline.add_frames_callback(stage, partial(self.board.set_frames, stage))
            self.pipeline.add_group_callback(stage, partial(self.board.set_group, stage))
            self.board.set_window_source(stage, self.pipeline.window(stage))
        self.pipeline.add_frames_callback(Stage.LERP, partial(self.osc_sound_sender.submit_frames, int(Stage.LERP)))
        self.pipeline.add_frames_callback(Stage.RAW, self._record_poses)     # the RAW stage's poses, with the video

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
        self.udp_light_receiver.bind("/WS/sensor/fall", self.conductor.set_fall)
        # Sound levels from Max (one 0..1 float each) → board → the beam_blue_sound layer.
        self.osc_sound_receiver.bind("/WS/idle/blue/left",  self._on_sound_level_left)
        self.osc_sound_receiver.bind("/WS/idle/blue/right", self._on_sound_level_right)
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
        self.state_machine.add_state_callback(self.osc_sound_sender.set_sequencer_state)

        # CAMERA HEALTH — compares each camera's own IMU reading and frame rate against the preset,
        # and summarises them as pinned indicators. A setup aid; nothing downstream reads it.
        self.camera_health = CameraHealth(self.settings.camera.cameras, self.settings.camera.camera_health)

        # RENDER
        self.render = WhiteSpaceRender(self.board, self.settings.render, self.settings.tracker,
                                       self.settings.camera.cameras, self.settings.camera.camera_health, ps.angle_extractor)
        self.render.add_update_callback(self.camera_health.update)
        self.render.add_update_callback(self._update_drops)
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
        self.pipeline.start()
        self.frame_source.start()
        self.uploader.start()
        self.frame_sync.start()
        self.tracklet_sync.start()
        for camera in self.cameras:
            camera.start()

        self.tracker.start()
        self.pose_estimator.start()
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

    def _on_tracklet_set(self, items: dict[int, CameraTracklets]) -> None:
        """A set of the cameras' tracklet messages, on the tracklets' sync's thread: the tracker's update, at the newest
        of their capture times."""
        self.tracker.update(max(message.timestamp for message in items.values()))

    def _on_asked(self, timestamp: float, cam_id: int, asked: FrameDict) -> None:
        """A set's poses for one camera, on the frame source's thread: the recorder hears whether poses come for the
        camera's image (the pose model asked, or the camera's own keypoints handed on), so a frame without a player
        does not wait."""
        posing: bool = self.settings.pose.pose.enabled or any(has_points(frame) for frame in asked.values())
        self.video_recorder.expect_poses(cam_id, timestamp, bool(asked) and posing)

    def _record_poses(self, frames: FrameDict) -> None:
        """The RAW stage's poses, on the pose model's thread, to the recorder as they are: their hand-over, stamped
        here; the recorder packs them on its own thread (`frame_poses`)."""
        self.video_recorder.submit_poses(frames, time.perf_counter())

    def _update_drops(self, _now: float) -> None:
        """The panel's drops, counted where they happen (the frame source's are on its own panel)."""
        cadence = self.settings.cadence
        cadence.frame_sync_drops = self.frame_sync.dropped
        cadence.tracklet_sync_drops = self.tracklet_sync.dropped
        cadence.upload_drops = self.uploader.dropped
        cadence.pose_drops = self.pipeline.dropped

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
        self.frame_sync.stop()
        self.tracklet_sync.stop()
        self.uploader.stop()
        self.frame_source.stop()
        self.pipeline.stop()
        self.video_recorder.stop()

        self.tracker.stop()
        self.tracker_journal.stop()
        self.osc_sound_sender.stop()
        self.state_machine.stop()
        self.osc_light_sender.stop()
        self.osc_sound_receiver.stop()
        self.udp_light_receiver.stop()
        self.conductor.stop()

        self.pose_estimator.stop()
        if self.person_detector is not None:
            self.person_detector.stop()

        for camera in self.cameras:
            camera.join(timeout=10)

        self.is_finished = True
