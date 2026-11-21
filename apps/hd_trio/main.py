"""HD Trio — 3-camera interactive installation with fluid rendering."""

import logging
import time
from functools import partial

from modules.camera import CameraFrame, CameraHealth, CameraSource, CameraTracklets
from modules.camera.oak import Camera, ClipCamera
from modules.recording import Player, Recorder as VideoRecorder, replay
from modules.settings import SettingsStore
from modules.panel import PanelServer
from modules.inout import OscSound, ArtNetBars, OscReceiver
from modules.tracker import OnePerCamTracker, TrackerFrameSource
from modules.pose import nodes, features, analytics, FrameDict, has_points, Recorder as PoseRecorder, RecorderSettings as PoseRecorderSettings
from modules.pose.pipeline import Domain, PosePipeline
from modules.inference import pose, segmentation, person, BatchUploader, CameraSync, CropExtractor
from modules.session import Session, SessionSettings, SequencerState
from modules.window import WindowManagerSettings

from .board import RenderBoard
from .pose.pipeline import pipeline_spec
from .sequencer import Sequencer
from .settings import Settings, Stage, ShowStage, IntroSequenceSettings
from .render import HDTrioRender

logger = logging.getLogger(__name__)

DATA_PATH = 'apps/hd_trio/data'


class HDTrioMain:
    def __init__(self, simulation: bool = False, interface: bool = True) -> None:
        self.is_running: bool = False
        self.is_finished: bool = False

        # SETTINGS
        self.settings = Settings()
        self.store = SettingsStore(f'{DATA_PATH}/settings')
        self.store.load_startup(self.settings)
        self.settings.setup.camera.sim_enabled = simulation
        if not interface:
            self.settings.setup.panel.browser = False  # launcher override; the settings file cannot turn it back on
        self.settings.setup.render.intro_sequence.arm_record = False     # never armed at start (HD_TRIO.md, The intro figure)
        pose_crop = self.settings.setup.pose.pose_crop                   # cut at the pose model's input size (HD_TRIO.md, Input)
        pose_crop.output_width = self.settings.setup.pose.pose.width
        pose_crop.output_height = self.settings.setup.pose.pose.height
        self.settings.initialize()
        self.panel_server = PanelServer(self.settings, self.settings.setup.panel, self.store, on_exit=self.stop)

        max_players: int = self.settings.setup.max_players
        num_cams: int = self.settings.setup.num_cams
        ps = self.settings.setup.pose

        # BLACKBOARD
        self.board = RenderBoard()

        # SESSION
        self.session = Session(self.settings.setup.session.control)
        self.session_osc = OscReceiver(self.settings.setup.session.osc)
        self.session_osc.bind('/start/recording', self._on_osc_start_recording)
        self.session_osc.bind('/stop/recording',  self._on_osc_stop_recording)
        self.session_osc.bind('/group/id',        self._on_osc_group_id)
        self.osc_sound = OscSound(self.settings.setup.inout.osc_sound)
        self.sequencer = Sequencer(self.settings.setup.session.sequencer)
        self.sequencer.add_state_callback(self.board.set_sequencer_state)
        self.sequencer.add_state_callback(self.osc_sound.set_sequencer_state)
        self.video_recorder = VideoRecorder(self.settings.setup.session.video, data_path=DATA_PATH)
        self.pose_recorder = PoseRecorder(self.settings.setup.session.pose, data_path=DATA_PATH)

        # The intro figure's recorder, outside the session's shared start and stop (HD_TRIO.md, The intro figure)
        self.intro_recorder = PoseRecorder(self.settings.setup.render.intro_sequence.recorder, data_path=DATA_PATH)
        self._prev_in_record_window: bool = False
        self.sequencer.add_state_callback(self._on_show_state)
        self.settings.setup.render.intro_sequence.bind(IntroSequenceSettings.arm_record, self._on_arm_record)
        self.artnet_controllers: list[ArtNetBars] = []
        for i in range(max_players):
            self.artnet_controllers.append(ArtNetBars(self.settings.setup.inout.artnets[i]))

        # CAMERA
        self.cameras: list[CameraSource] = []
        self.player: Player | None = None
        self.person_detector: person.PersonDetector | None = None      # the replay backend's, shared by its cameras
        if self.settings.setup.camera.sim_enabled:
            self.player = Player(self.settings.setup.camera.simulator, data_path=DATA_PATH)
            if self.settings.setup.camera.simulator.replay:
                # the recordings on the host, without a device: the people found and followed here
                if self.settings.setup.camera.yolo:
                    self.person_detector = person.PersonDetector(self.settings.setup.camera.person_detector)
                for i in range(num_cams):
                    self.cameras.append(replay.Camera(self.player, self.settings.setup.camera.cameras[i], self.person_detector, replay.Tracker(), i))
            else:
                for i in range(num_cams):
                    self.cameras.append(ClipCamera(self.player, self.settings.setup.camera.cameras[i]))
        else:
            for i in range(num_cams):
                self.cameras.append(Camera(self.settings.setup.camera.cameras[i]))
        self.frame_sync: CameraSync[CameraFrame] = CameraSync(self.settings.setup.camera.frame_sync, 'frame_sync')
        self.uploader = BatchUploader()          # a synced set of the cameras' frames onto the GPU as one batch, on its own thread
        self.tracker = OnePerCamTracker(self.settings.setup.camera.tracker, max_players)
        self.tracklet_sync: CameraSync[CameraTracklets] = CameraSync(self.settings.setup.camera.tracklet_sync, 'tracklet_sync')
        # The frame domain's async step: the crop step in (the pose model's crops and the segmentation's, in one pass,
        # from the same images and boxes), the pose model out
        self.crops = CropExtractor(ps.pose_crop, ps.segmentation_crop)
        self.pose_estimator = pose.PoseEstimator(ps.pose)
        self.crops.add_image_callback(self.pose_estimator.submit)

        # The camera's thread only hands its frames and tracklets over: the frames to the frames' sync and the
        # recorder, the tracklets as they come to the tracker, the board, the tracklets' sync and the recorder
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
        device_tracklets: bool = not (self.settings.setup.camera.sim_enabled and self.settings.setup.camera.simulator.replay)
        for i in range(num_cams):
            self.video_recorder.expect_parts(i, tracklets=device_tracklets, poses=True)

        # DETECTION
        features.configure_features(max_players)

        self.frame_source = TrackerFrameSource(ps.frame_source, max_players)
        self.segmenter = segmentation.Segmenter(ps.segmentation)

        self.tracker.add_tracklet_callback(self.frame_source.set_tracklets)
        # The frame source's thread: each set's poses with its images to the pipeline; the crop step in it then
        # tells the poses asked per camera
        self.crops.add_asked_callback(self._on_asked)

        # THE SYNCS' HAND-OFFS, each on its own thread: the frames' set to the uploader, the tracklets' to the tracker
        self.frame_sync.add_set_callback(self.uploader.submit_set)
        self.tracklet_sync.add_set_callback(self._on_tracklet_set)

        # THE UPLOADER'S HAND-OFFS, in order on its thread: each uploaded image to the detector (-sim replay) and the
        # board, then the set's snapshot to the frame source
        if self.person_detector is not None:
            self.uploader.add_gpu_image_callback(self.person_detector.submit)
        self.uploader.add_finished_gpu_image_callback(self.board.set_camera_image)
        self.uploader.add_snapshot_callback(self.crops.set_snapshot)
        self.uploader.add_capture_callback(self.frame_source.submit)

        # The segmentation and the board read the segmentation's crops, cut from the same images and boxes as the pose model's
        self.crops.add_image_callback(self.segmenter.submit, ps.segmentation_crop)
        self.crops.add_finished_image_callback(lambda _f, crops: self.board.set_crop_images(crops), ps.segmentation_crop)
        self.segmenter.add_segmentation_callback(lambda _f, masks: self.board.set_segmentation_images(masks))

        # POSE PIPELINE — the stages are declared in pose/pipeline.py. Each uploaded set drives the frame domain
        # through the tracklet poses (the uploader's capture times);
        # the render tick drives the output domain (LERP). Every stage's
        # frames go to the board, the pose recorders and the sound; the board also reads each stage's window; the RAW
        # stage's poses go to the video recorder with their frames.
        self.similarity_hold = nodes.SimilarityResultHold()              # the threaded analytics' way back into SMOOTH
        self.pipeline = PosePipeline(pipeline_spec(ps, self.frame_source, self.crops, self.pose_estimator, self.similarity_hold))
        for stage in Stage:
            self.pipeline.add_frames_callback(stage, partial(self.board.set_frames, stage))
            self.pipeline.add_frames_callback(stage, partial(self.pose_recorder.submit_frames, stage))
            self.pipeline.add_frames_callback(stage, partial(self._submit_intro_frames, stage))
            self.pipeline.add_frames_callback(stage, partial(self.osc_sound.submit_frames, stage))
            self.board.set_window_source(stage, self.pipeline.window(stage))
        self.pipeline.add_frames_callback(Stage.RAW, self._record_poses)

        # The threaded analytics read the SMOOTH window and hand their result to the hold.
        self.window_similarity  = analytics.WindowSimilarity(ps.similarity.window_similarity)
        self.window_correlator = analytics.WindowCorrelation(ps.similarity.window_correlation)
        self.pipeline.window(Stage.SMOOTH).add_windows_callback(self.window_similarity.submit)
        self.window_similarity.add_similarity_callback(self.similarity_hold.set)
        self.pipeline.window(Stage.SMOOTH).add_windows_callback(self.window_correlator.submit)
        self.window_correlator.add_similarity_callback(self.similarity_hold.set)

        # CAMERA HEALTH — each camera's IMU reading and frame rate against the settings file, as pinned indicators
        self.camera_health = CameraHealth(self.settings.setup.camera.cameras, self.settings.setup.camera.camera_health)

        # RENDER
        self.render = HDTrioRender(self.board, self.settings.setup.render, self.settings.setup.windows, self.settings.setup.camera.cameras,
                                   self.settings.setup.camera.camera_health, data_path=DATA_PATH)
        self.settings.setup.windows.bind(WindowManagerSettings.avg_fps, self._on_render_fps)
        self.render.add_update_callback(self.camera_health.update)
        self.render.add_update_callback(self._update_drops)
        self.render.add_update_callback(self.sequencer.update)
        self.render.add_update_callback(self._update_poses)
        self.render.add_exit_callback(self.stop)

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
        self.segmenter.start()
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

    def _on_tracklet_set(self, items: dict[int, CameraTracklets]) -> None:
        """A set of the cameras' tracklet messages, on the tracklets' sync's thread: the tracker's update, at the newest
        of their capture times."""
        self.tracker.update(max(message.timestamp for message in items.values()))

    def _on_asked(self, timestamp: float, cam_id: int, asked: FrameDict) -> None:
        """A set's poses for one camera, on the frame source's thread: the recorder hears whether poses come for the
        camera's image, so a frame without a player does not wait."""
        posing: bool = self.settings.setup.pose.pose.enabled or any(has_points(frame) for frame in asked.values())
        self.video_recorder.expect_poses(cam_id, timestamp, bool(asked) and posing)

    def _record_poses(self, frames: FrameDict) -> None:
        """The RAW stage's poses, on the pose model's thread, to the recorder as they are: their hand-over, stamped
        here; the recorder packs them on its own thread."""
        self.video_recorder.submit_poses(frames, time.perf_counter())

    def _update_drops(self, _now: float) -> None:
        """The panel's drops, counted where they happen (the frame source's are on its own panel)."""
        cadence = self.settings.setup.cadence
        cadence.frame_sync_drops = self.frame_sync.dropped
        cadence.tracklet_sync_drops = self.tracklet_sync.dropped
        cadence.upload_drops = self.uploader.dropped
        cadence.pose_drops = self.pipeline.dropped

    def _update_poses(self, now: float) -> None:
        """The render tick drives the pipeline's output domain (LERP) at the frame's time."""
        self.pipeline.update(Domain.OUTPUT, now)

    def stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False

        self.panel_server.stop()
        self.settings.setup.render.intro_sequence.unbind(IntroSequenceSettings.arm_record, self._on_arm_record)
        self.settings.setup.windows.unbind(WindowManagerSettings.avg_fps, self._on_render_fps)
        self.sequencer.stop()

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
        # a pose recorder writes its chunk on its stop button only
        if self.settings.setup.render.intro_sequence.recorder.recording:
            PoseRecorderSettings.stop.fire(self.settings.setup.render.intro_sequence.recorder)

        self.tracker.stop()
        self.osc_sound.stop()

        for artnet in self.artnet_controllers:
            artnet.stop()

        self.session_osc.stop()

        self.pose_estimator.stop()
        if self.person_detector is not None:
            self.person_detector.stop()
        self.segmenter.stop()
        self.window_similarity.stop()
        self.window_correlator.stop()

        for camera in self.cameras:
            camera.join(timeout=10)

        self.is_finished = True

    def _on_arm_record(self, armed: bool) -> None:
        """Warn at arm time if the configured window can never open."""
        intro = self.settings.setup.render.intro_sequence
        if armed and intro.record_end < intro.record_start:
            logger.warning("intro record window is empty: record_end (%s) is before record_start (%s)",
                           intro.record_end.name, intro.record_start.name)

    def _submit_intro_frames(self, stage: Stage, frames: FrameDict) -> None:
        """Submit only the performer's track, so the intro file holds one figure."""
        track_id: int = self.settings.setup.render.intro_sequence.record_track
        frame = frames.get(track_id)
        self.intro_recorder.submit_frames(stage, {track_id: frame} if frame is not None else {})

    def _on_show_state(self, state: SequencerState) -> None:
        """Start and stop the intro recording on its stage window, `record_start` to `record_end` in show order."""
        intro = self.settings.setup.render.intro_sequence
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
        SessionSettings.start.fire(self.settings.setup.session)

    def _on_osc_stop_recording(self, *_) -> None:
        SessionSettings.stop.fire(self.settings.setup.session)

    def _on_osc_group_id(self, gid: str, *_) -> None:
        self.settings.setup.session.name = gid

    def _on_render_fps(self, fps: int) -> None:
        if fps > 0:
            self.settings.setup.render_fps = float(fps)
