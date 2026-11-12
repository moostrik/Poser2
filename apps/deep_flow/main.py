"""Deep Flow — single-camera 3D volumetric fluid installation.

Runs in `.venv-oak4`, the depthai 3 venv (`scripts\\install.bat oak4`), on one OAK4. `-sim` plays the
recordings on the host with the replay backend, with no camera attached.
"""

import time
from functools import partial

from modules.camera import CameraSource, CameraCheck, CameraTracklet, Player, Recorder, replay
from modules.camera.oak4 import Camera, ModelChain
from modules.session import Session
from modules.settings import presets
from modules.panel import PanelServer
from modules.inout import OscSound
from modules.tracker import OnePerCamTracker, PosesFromTracklets
from modules.pose import configure_features
from modules.pose.pipeline import Domain, PosePipeline
from modules.inference import optical_flow, detection, GpuImage, PoseInference, Capture, CaptureSync, Uploader
from modules.pose import FrameDict
from modules.utils import FPS, PerformanceTimer

from .render_board import RenderBoard
from .series import Series, TIMER_SERIES
from .pose.pipeline import pipeline_spec
from .settings import Settings, Stage
from .render import DeepFlowRender

APP_NAME = 'deep_flow'
DATA_PATH = 'apps/deep_flow/data'


class DeepFlowMain:
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
        self.settings.initialize()
        self.panel_server = PanelServer(self.settings, self.settings.panel, on_exit=self.stop)

        max_players: int = self.settings.max_players
        num_cams: int = self.settings.num_cams
        p = self.settings.pose

        # BLACKBOARD
        self.board = RenderBoard()
        self.osc_sound = OscSound(self.settings.inout.osc_sound)

        # CAMERA
        configure_features(max_players)

        self.cameras: list[CameraSource] = []
        self.recorder: Recorder | None = None
        self.player: Player | None = None
        self.person_detector: detection.PersonDetector | None = None      # the replay backend's, shared by its cameras
        if self.settings.camera.sim_enabled:
            self.player = Player(self.settings.camera.simulator, data_path=DATA_PATH)
            # the recordings on the host, without a device: the people found and followed here
            if self.settings.camera.cam_0.model_chain != ModelChain.OFF:
                self.person_detector = detection.PersonDetector(self.settings.camera.person_detector)
            for i in range(num_cams):
                cam = self.settings.camera.cameras[i]
                self.cameras.append(replay.Camera(self.player, cam, self.person_detector, replay.Tracker(), i, cam.tilt))
        else:
            self.recorder = Recorder(self.settings.camera.recorder, data_path=DATA_PATH)
            self.session = Session(self.settings.camera.control)       # splits the recording into chunks
            for i in range(num_cams):
                self.cameras.append(Camera(self.settings.camera.cameras[i]))
        self.tracker = OnePerCamTracker(self.settings.camera.tracker, max_players)
        self.uploader = Uploader()                                       # the camera's hand-off: the frames onto the GPU, on its own thread
        self.pose_inference = PoseInference(p.image_crop, p.pose, self.uploader)    # the crop and the pose model, the frame domain's async step
        self.capture_sync = CaptureSync(p.capture_sync)                  # the color, depth and poses of one moment, for the render
        self.capture_rate: FPS = FPS(120)                                # the captures it hands on, for the panel's cadence

        # The camera's thread only hands its frames over: to the recorder as they came, to the uploader for the rest
        for camera in self.cameras:
            if self.recorder:
                camera.add_frame_callback(self.recorder.submit_frame)
            camera.add_frame_callback(self.uploader.submit_frame)
            camera.add_tracklet_callback(self.tracker.submit_cam_tracklets)
            camera.add_tracklet_callback(self.board.set_camera_tracklets)
            camera.add_tracklet_callback(self._on_tracklets)

        # DETECTION
        self.poses_from_tracklets = PosesFromTracklets(p.tracklets, max_players)
        self.optical_flow_predictor  = optical_flow.Predictor(p.optical_flow)

        self.tracker.add_tracklet_callback(self.poses_from_tracklets.set_tracklets)

        # The optical flow reads the same crops the pose model gets.
        self.pose_inference.add_crop_callback(self.optical_flow_predictor.submit)

        # POSE PIPELINE — the stages are declared in pose/pipeline.py. Each uploaded image drives the frame domain
        # through the tracklet poses (`_on_image`); the render tick drives the output domain (LERP). Every stage's frames
        # go to the board and the sound; the board also reads each stage's window.
        self.pipeline = PosePipeline(pipeline_spec(p, self.poses_from_tracklets, self.pose_inference))
        for stage in Stage:
            self.pipeline.add_frames_callback(stage, partial(self.board.set_frames, stage))
            self.pipeline.add_frames_callback(stage, partial(self.osc_sound.submit_frames, stage))
            self.board.set_window_source(stage, self.pipeline.window(stage))
        self.pipeline.add_frames_callback(Stage.RAW, partial(self._on_pose_frames, Series.POSE_LATENCY_RAW))
        self.pipeline.add_frames_callback(Stage.SMOOTH, partial(self._on_pose_frames, Series.POSE_LATENCY_SMOOTH))

        # THE UPLOADER'S HAND-OFFS, in order on its thread: each uploaded image to the detector (-sim), the capture
        # sync and the pose step, each uploaded depth to the capture sync
        if self.person_detector is not None:
            self.uploader.add_gpu_image_callback(self.person_detector.submit)
        self.uploader.add_gpu_image_callback(self.capture_sync.submit_image)
        self.uploader.add_gpu_image_callback(self.board.set_camera_image)
        self.uploader.add_gpu_image_callback(self._on_image)
        self.uploader.add_gpu_depth_callback(self.capture_sync.submit_depth)

        # CAPTURE SYNC — each uploaded image, its depth and its SMOOTH poses, as one capture on the board
        self.pipeline.add_frames_callback(Stage.SMOOTH, self.capture_sync.submit_poses)
        self.capture_sync.add_capture_callback(self._on_capture)

        # CAMERA CHECK — the camera's IMU reading and frame rate against the preset, as pinned indicators
        self.camera_check = CameraCheck(self.settings.camera.cameras, self.settings.camera.camera_check)

        # RENDER
        self.render = DeepFlowRender(self.board, self.settings.render, self.settings.camera.fov, num_cams=len(self.cameras), max_players=max_players)
        self.render.add_exit_callback(self.stop)
        self.render.add_update_callback(self.camera_check.update)

        self.render.add_update_callback(self._update_poses)
        self.render.add_update_callback(self._update_cadence)

        # Every loop of a recording starts on clear water, so two runs of one clip compare
        if self.player:
            self.player.add_loop_callback(self.render.reset_fluid)

        # TIME SERIES — the timings the time series layer plots
        self._last_camera_frame: dict[int, float] = {}
        PerformanceTimer.add_sample_callback(self._on_timer_sample)

    def start(self) -> None:
        self.panel_server.start()

        if self.person_detector is not None:
            self.person_detector.start()
        self.uploader.start()
        for camera in self.cameras:
            camera.start()

        self.tracker.start()
        self.pose_inference.start()
        self.capture_sync.start()
        self.optical_flow_predictor.start()

        self.osc_sound.start()

        if self.player:
            self.player.start()
        if self.recorder:
            self.recorder.start()

        self.is_running = True
        self.render.start()

    def _on_image(self, image: GpuImage) -> None:
        """An uploaded image, on the uploader's thread: the pose step for it, and the camera's interval. With one
        camera every image is a round, so the frame domain runs on each, at the image's capture time."""
        self.poses_from_tracklets.process(image.timestamp)
        last: float | None = self._last_camera_frame.get(image.cam_id)
        self._last_camera_frame[image.cam_id] = image.timestamp
        if last is not None:
            self.board.submit_time_sample(Series.CAMERA_INTERVAL, (image.timestamp - last) * 1000.0, image.timestamp)

    def _on_tracklets(self, cam_id: int, tracklets: list[CameraTracklet], timestamp: float) -> None:
        """The camera's tracklets have reached the tracker's queue: its update, at their capture time."""
        self.tracker.update(timestamp)

    def _on_capture(self, capture: Capture) -> None:
        self.capture_rate.processed()
        self.board.set_capture(capture)
        now: float = time.perf_counter()        # when the capture reaches the board: the end of the latency
        self.board.submit_time_sample(Series.CAPTURE_LATENCY, (now - capture.timestamp) * 1000.0, now)
        if capture.poses:
            self.board.submit_time_sample(Series.POSES_UNSYNCED, 0.0 if capture.poses_synced else 1.0, now)

    def _on_pose_frames(self, series: Series, poses: FrameDict) -> None:
        """A stage's poses as they come out: how long after their capture."""
        if not poses:
            return
        now: float = time.perf_counter()        # when the poses come out of the stage: the end of the latency
        newest: float = max(frame.timestamp for frame in poses.values())
        self.board.submit_time_sample(series, (now - newest) * 1000.0, now)

    def _on_timer_sample(self, name: str, time_ms: float, timestamp: float) -> None:
        series: Series | None = TIMER_SERIES.get(name)
        if series is not None:
            self.board.submit_time_sample(series, time_ms, timestamp)

    def _update_cadence(self, now: float) -> None:
        """The panel's cadence: the captures the sync hands on and the frames the motion measures, the slowest camera's."""
        cadence = self.settings.cadence
        cadence.capture_fps = round(self.capture_rate.get_rate_average(), 1)
        cadence.capture_min_fps = round(self.capture_rate.get_rate_minimum(), 1)
        rates = list(self.render.motion_rates.values())
        cadence.motion_fps = round(min((rate.get_rate_average() for rate in rates), default=0.0), 1)
        cadence.motion_min_fps = round(min((rate.get_rate_minimum() for rate in rates), default=0.0), 1)

    def _update_poses(self, now: float) -> None:
        """The render tick drives the pipeline's output domain (LERP) at the frame's time."""
        self.pipeline.update(Domain.OUTPUT, now)

    def stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False

        PerformanceTimer.remove_sample_callback(self._on_timer_sample)
        self.panel_server.stop()

        self.render.stop()

        if self.player:
            self.player.stop()
        for camera in self.cameras:
            camera.stop()
        self.uploader.stop()
        if self.recorder:
            self.recorder.stop()

        self.tracker.stop()
        self.osc_sound.stop()

        self.pose_inference.stop()
        self.capture_sync.stop()
        if self.person_detector is not None:
            self.person_detector.stop()
        self.optical_flow_predictor.stop()

        for camera in self.cameras:
            camera.join(timeout=10)

        self.is_finished = True
