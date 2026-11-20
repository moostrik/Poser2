"""Low Flow — single-camera 2D fluid installation.

Runs in `.venv-oak4`, the depthai 3 venv (`scripts\\install.bat oak4`), on one OAK4, without depth. `-sim` plays the
recordings' video on the host with the replay backend, with no camera attached.
"""

import time
from functools import partial

from modules.camera import CameraSource, CameraHealth, CameraTracklets
from modules.camera.oak4 import Camera, ModelChain
from modules.recording import Player, Recorder, replay
from modules.session import Session
from modules.settings import presets
from modules.panel import PanelServer
from modules.inout import OscSound
from modules.tracker import OnePerCamTracker, TrackerFrameSource
from modules.pose import configure_features
from modules.pose.pipeline import Domain, PosePipeline
from modules.inference import pose, person, segmentation, CropExtractor, GpuImage, has_points, RecordedPoses, InferenceFrame, InferenceFrameSync, Masks, SerialUploader, frame_poses
from modules.pose import FrameDict
from modules.utils import FPS, PerformanceTimer

from .board import RenderBoard
from .series import Series, TIMER_SERIES
from .pose.pipeline import pipeline_spec
from .settings import Settings, Stage
from .render import LowFlowRender

APP_NAME = 'low_flow'
DATA_PATH = 'apps/low_flow/data'


class LowFlowMain:
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
        # from the model so a change of pose model cannot silently reintroduce the second one.
        crop_settings = self.settings.pose.pose_crop
        crop_settings.output_width = self.settings.pose.pose.width
        crop_settings.output_height = self.settings.pose.pose.height
        # The segmentation's crop likewise, at the segmentation model's size
        segmentation_crop = self.settings.pose.segmentation_crop
        segmentation_crop.output_width = self.settings.pose.segmentation.width
        segmentation_crop.output_height = self.settings.pose.segmentation.height
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
        self.player: Player | None = None
        # Live and in -sim: a recording played can be recorded again, with what the app found in it (its poses)
        self.recorder: Recorder = Recorder(self.settings.camera.recorder, frame_poses, data_path=DATA_PATH)
        self.session = Session(self.settings.camera.control)       # splits the recording into chunks
        self.person_detector: person.PersonDetector | None = None      # the replay backend's, shared by its cameras
        if self.settings.camera.sim_enabled:
            self.player = Player(self.settings.camera.simulator, data_path=DATA_PATH, nv12=True)     # as the camera hands its frames
            # the recordings on the host, without a device: the people found and followed here
            if self.settings.camera.cam_0.model_chain != ModelChain.OFF:
                self.person_detector = person.PersonDetector(self.settings.camera.person_detector)
            for i in range(num_cams):
                cam = self.settings.camera.cameras[i]
                self.cameras.append(replay.Camera(self.player, cam, self.person_detector, replay.Tracker(), i, cam.tilt))
        else:
            for i in range(num_cams):
                self.cameras.append(Camera(self.settings.camera.cameras[i]))
        self.tracker = OnePerCamTracker(self.settings.camera.tracker, max_players)
        self.uploader = SerialUploader(self.settings.camera.uploader)    # the camera's hand-off: each frame onto the GPU as it comes, on its own thread
        # The frame domain's async step: the crop step in (the pose model's crops at their interval and the
        # segmentation's every frame, in one pass, from the same images and boxes), the pose model out
        self.crops = CropExtractor(p.pose_crop, p.segmentation_crop)
        self.pose_estimator = pose.PoseEstimator(p.pose)
        # The segmentation, every frame, on its own crops
        self.segmenter = segmentation.Segmenter(p.segmentation)
        self.crops.add_image_callback(self.segmenter.submit, p.segmentation_crop)
        # -sim plays the recording's poses in place of the pose model, between the crops and it (`recorded_poses.enabled`)
        self.recorded_poses: RecordedPoses | None = None
        poses: pose.PoseEstimator | RecordedPoses = self.pose_estimator
        if self.player is not None:
            self.recorded_poses = RecordedPoses(self.settings.camera.recorded_poses, self.pose_estimator)
            self.player.add_frame_callback(self.recorded_poses.set_clip_frame)
            self.player.add_jump_callback(self.recorded_poses.clear_waiting)
            poses = self.recorded_poses
        self.crops.add_image_callback(poses.submit)
        # the image and the mask of one moment, for the render; the poses are not in it: the render takes the
        # interpolated poses at its tick
        self.inference_frame_sync = InferenceFrameSync(p.inference_frame_sync, (Masks,))
        self.inference_frame_rate: FPS = FPS()                                # the frames it hands on, for the panel's cadence

        # The camera's thread only hands its frames and tracklets over: to the recorder as they came, to the uploader
        # for the rest, which straightens the tracklets with their frame and hands them to the tracker and the board
        for camera in self.cameras:
            camera.add_frame_callback(self.uploader.submit_frame)           # the live path first, the recorder after it
            camera.add_tracklet_callback(self.uploader.submit_tracklets)
            camera.add_frame_callback(self.recorder.submit_frame)
            camera.add_tracklet_callback(self.recorder.submit_tracklets)
        # What each camera's frames wait for from the first: live, a camera with a model sends tracklets (a replay's
        # come or not with the recording); the crop step says per image whether it asked poses (`_on_asked`)
        live_tracklets: bool = not self.settings.camera.sim_enabled and self.settings.camera.cam_0.model_chain != ModelChain.OFF
        for i in range(num_cams):
            self.recorder.expect_parts(i, tracklets=live_tracklets, poses=True)
        self.uploader.add_tracklet_callback(self.tracker.submit_cam_tracklets)
        self.uploader.add_tracklet_callback(self.board.set_camera_tracklets)
        self.uploader.add_tracklet_callback(self._on_tracklets)

        # DETECTION
        self.frame_source = TrackerFrameSource(p.frame_source, max_players)

        self.tracker.add_tracklet_callback(self.frame_source.set_tracklets)
        # The frame source's thread: each bang's poses with its images to the pipeline; the crop step in it then
        # tells the poses asked (to the recorder) and the masks asked (to the inference frame sync)
        self.crops.add_asked_callback(self._on_asked)
        self.crops.add_asked_callback(self._on_masks_asked, p.segmentation_crop)

        # POSE PIPELINE — the stages are declared in pose/pipeline.py. Each uploaded image drives the frame domain
        # through the tracklet poses (the uploader's capture times);
        # the render tick drives the output domain (LERP). Every stage's frames
        # go to the board and the sound; the board also reads each stage's window.
        self.pipeline = PosePipeline(pipeline_spec(p, self.frame_source, self.crops, poses))
        for stage in Stage:
            self.pipeline.add_frames_callback(stage, partial(self.board.set_frames, stage))
            self.pipeline.add_frames_callback(stage, partial(self.osc_sound.submit_frames, stage))
            self.board.set_window_source(stage, self.pipeline.window(stage))
        self.pipeline.add_frames_callback(Stage.RAW, partial(self._on_pose_frames, Series.POSE_LATENCY_RAW))
        self.pipeline.add_frames_callback(Stage.RAW, self._record_poses)     # the RAW stage's poses, with the video
        self.pipeline.add_frames_callback(Stage.SMOOTH, partial(self._on_pose_frames, Series.POSE_LATENCY_SMOOTH))

        # THE UPLOADER'S HAND-OFFS, in order on its thread: each uploaded image to the detector (-sim), the inference
        # frame sync, the board and the camera's interval, then its snapshot to the frame source
        if self.person_detector is not None:
            self.uploader.add_gpu_image_callback(self.person_detector.submit)
        self.uploader.add_finished_gpu_image_callback(self.inference_frame_sync.submit_image)
        self.uploader.add_finished_gpu_image_callback(self.board.set_camera_image)
        self.uploader.add_finished_gpu_image_callback(self._on_image)
        self.uploader.add_snapshot_callback(self.crops.submit_snapshot)
        self.uploader.add_capture_callback(self.frame_source.submit)

        # INFERENCE FRAME SYNC — each uploaded image and its masks, as one frame on the board
        self.segmenter.add_segmentation_callback(self._on_masks)
        self.inference_frame_sync.add_frame_callback(self._on_inference_frame)

        # CAMERA HEALTH — the camera's IMU reading and frame rate against the preset, as pinned indicators
        self.camera_health = CameraHealth(self.settings.camera.cameras, self.settings.camera.camera_health)

        # RENDER
        self.render = LowFlowRender(self.board, self.settings.render, self.settings.screen_recorder, f'{DATA_PATH}/screen_recordings',
                                            num_cams=len(self.cameras), max_players=max_players)
        self.render.add_exit_callback(self.stop)
        self.render.add_update_callback(self.camera_health.update)

        self.render.add_update_callback(self._update_poses)
        self.render.add_update_callback(self._update_cadence)

        # Every loop of a recording starts the flow from rest, so two runs of one clip compare
        if self.player:
            self.player.add_loop_callback(self.render.reset_flow)

        # TIME SERIES — the timings the time series layer plots
        self._last_camera_frame: dict[int, float] = {}
        PerformanceTimer.add_sample_callback(self._on_timer_sample)

    def start(self) -> None:
        self.panel_server.start()

        if self.person_detector is not None:
            self.person_detector.start()
        self.pipeline.start()
        self.frame_source.start()
        self.inference_frame_sync.start()
        self.uploader.start()
        for camera in self.cameras:
            camera.start()

        self.tracker.start()
        if self.recorded_poses is not None:
            self.recorded_poses.start()         # starts the pose model with it
        else:
            self.pose_estimator.start()
        self.segmenter.start()

        self.osc_sound.start()

        if self.player:
            self.player.start()
        self.recorder.start()

        self.is_running = True
        self.render.start()

    def _on_image(self, image: GpuImage) -> None:
        """An uploaded image, on the uploader's thread: the camera's interval."""
        last: float | None = self._last_camera_frame.get(image.cam_id)
        self._last_camera_frame[image.cam_id] = image.timestamp
        if last is not None:
            self.board.set_time_sample(Series.CAMERA_INTERVAL, (image.timestamp - last) * 1000.0, image.timestamp)

    def _on_asked(self, timestamp: float, cam_id: int, asked: FrameDict) -> None:
        """A frame bang's poses, on the frame source's thread: the recorder hears whether poses come for the image (the
        pose model asked, the recording's played, or the camera's own keypoints handed on), so a frame without a
        player, or passed by the pose crop's `interval`, does not wait."""
        modelled: bool = (self.recorded_poses is not None and self.settings.camera.recorded_poses.enabled) or self.settings.pose.pose.enabled
        posing: bool = modelled or any(has_points(frame) for frame in asked.values())
        self.recorder.expect_poses(cam_id, timestamp, bool(asked) and posing)

    def _on_masks_asked(self, timestamp: float, cam_id: int, asked: FrameDict) -> None:
        """A frame bang's segmentation crops, on the frame source's thread: the inference frame sync hears whether masks
        come for the image, so a frame without a player does not wait."""
        self.inference_frame_sync.expect(cam_id, timestamp, bool(asked) and self.settings.pose.segmentation.enabled)

    def _on_masks(self, poses: FrameDict, masks: segmentation.ImageDict) -> None:
        """The segmentation's masks, on its thread, to the inference frame sync, one result a camera."""
        self.inference_frame_sync.submit(*Masks.by_camera(poses, masks))

    def _record_poses(self, frames: FrameDict) -> None:
        """The RAW stage's poses, on the pose model's thread, to the recorder as they are: their hand-over, stamped
        here; the recorder packs them on its own thread (`frame_poses`)."""
        self.recorder.submit_poses(frames, time.perf_counter())

    def _on_tracklets(self, cam_tracklets: CameraTracklets) -> None:
        """The camera's tracklets have reached the tracker's queue: its update, at their capture time."""
        self.tracker.update(cam_tracklets.timestamp)

    def _on_inference_frame(self, frame: InferenceFrame) -> None:
        self.inference_frame_rate.processed()
        self.board.set_inference_frame(frame)
        now: float = time.perf_counter()        # when the frame reaches the board: the end of the latency
        self.board.set_time_sample(Series.CAPTURE_LATENCY, (now - frame.timestamp) * 1000.0, now)
        masks: Masks = frame[Masks]
        if masks.masks:
            self.board.set_time_sample(Series.MASKS_UNSYNCED, 0.0 if masks.synced else 1.0, now)

    def _on_pose_frames(self, series: Series, poses: FrameDict) -> None:
        """A stage's poses as they come out: how long after their capture."""
        if not poses:
            return
        now: float = time.perf_counter()        # when the poses come out of the stage: the end of the latency
        newest: float = max(frame.timestamp for frame in poses.values())
        self.board.set_time_sample(series, (now - newest) * 1000.0, now)

    def _on_timer_sample(self, name: str, time_ms: float, timestamp: float) -> None:
        series: Series | None = TIMER_SERIES.get(name)
        if series is not None:
            self.board.set_time_sample(series, time_ms, timestamp)

    def _update_cadence(self, now: float) -> None:
        """The panel's cadence: the frames the sync hands on and the frames the motion measures, the slowest camera's, and the drops."""
        cadence = self.settings.cadence
        cadence.inference_frame_fps = round(self.inference_frame_rate.get_rate_average(), 1)
        cadence.inference_frame_min_fps = round(self.inference_frame_rate.get_rate_minimum(), 1)
        rates = list(self.render.motion_rates.values())
        cadence.motion_fps = round(min((rate.get_rate_average() for rate in rates), default=0.0), 1)
        cadence.motion_min_fps = round(min((rate.get_rate_minimum() for rate in rates), default=0.0), 1)
        cadence.upload_drops = self.uploader.dropped
        cadence.inference_frame_drops = self.board.get_dropped_inference_frames()
        cadence.pose_drops = self.pipeline.dropped

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
        self.frame_source.stop()
        self.pipeline.stop()
        self.recorder.stop()

        self.tracker.stop()
        self.osc_sound.stop()

        if self.recorded_poses is not None:
            self.recorded_poses.stop()          # stops the pose model with it
        else:
            self.pose_estimator.stop()
        self.segmenter.stop()
        self.inference_frame_sync.stop()
        if self.person_detector is not None:
            self.person_detector.stop()

        for camera in self.cameras:
            camera.join(timeout=10)

        self.is_finished = True
