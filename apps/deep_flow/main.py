"""Deep Flow — single-camera 3D volumetric fluid installation."""

from functools import partial

from modules.camera import CameraSource, CameraCheck, Player, Recorder, Sync, replay
from modules.camera.oak import Camera, Simulator
from modules.settings import presets
from modules.panel import PanelServer
from modules.inout import OscSound
from modules.tracker import OnePerCamTracker, PosesFromTracklets
from modules.pose import configure_features
from modules.pose.pipeline import Domain, PosePipeline
from modules.inference import segmentation, optical_flow, detection, PoseInference

from .render_board import RenderBoard
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
            self.recorder = Recorder(self.settings.camera.recorder, data_path=DATA_PATH)
            for i in range(num_cams):
                self.cameras.append(Camera(self.settings.camera.cameras[i]))
        self.frame_sync_bang = Sync(self.settings.camera.frame_sync, False, 'frame_sync')
        self.tracker = OnePerCamTracker(self.settings.camera.tracker, max_players)
        self.tracklet_sync_bang = Sync(self.settings.camera.tracklet_sync, False, 'tracklet_sync')
        self.pose_inference = PoseInference(p.image_crop, p.pose)        # the crop and the pose model, the frame domain's async step

        for camera in self.cameras:
            if self.recorder:
                camera.add_frame_callback(self.recorder.submit_frame)
            camera.add_frame_callback(self.pose_inference.set_camera_frame)
            camera.add_frame_callback(self.frame_sync_bang.submit_frame)
            camera.add_tracklet_callback(self.tracker.submit_cam_tracklets)
            camera.add_tracklet_callback(self.board.set_camera_tracklets)
            camera.add_tracklet_callback(self.tracklet_sync_bang.submit_tracklets)

        # DETECTION
        self.poses_from_tracklets = PosesFromTracklets(p.tracklets, max_players)
        self.segmentation_predictor  = segmentation.Predictor(p.segmentation)
        self.optical_flow_predictor  = optical_flow.Predictor(p.optical_flow)

        self.tracker.add_tracklet_callback(self.poses_from_tracklets.set_tracklets)
        self.tracklet_sync_bang.add_sync_callback(self.tracker.update)
        self.frame_sync_bang.add_sync_callback(self.poses_from_tracklets.process)

        # The segmentation and the optical flow read the same crops the pose model gets.
        self.pose_inference.add_camera_images_callback(self.board.set_camera_images)
        self.pose_inference.add_crop_callback(self.segmentation_predictor.submit)
        self.pose_inference.add_crop_callback(self.optical_flow_predictor.submit)
        self.segmentation_predictor.add_segmentation_callback(lambda _f, masks: self.board.set_segmentation_images(masks))

        # POSE PIPELINE — the stages are declared in pose/pipeline.py. The camera's frame bang drives the frame
        # domain through the tracklet poses; the render tick drives the output domain (LERP). Every stage's frames
        # go to the board and the sound; the board also reads each stage's window.
        self.pipeline = PosePipeline(pipeline_spec(p, self.poses_from_tracklets, self.pose_inference))
        for stage in Stage:
            self.pipeline.add_frames_callback(stage, partial(self.board.set_frames, stage))
            self.pipeline.add_frames_callback(stage, partial(self.osc_sound.submit_frames, stage))
            self.board.set_window_source(stage, self.pipeline.window(stage))

        # CAMERA CHECK — the camera's IMU reading and frame rate against the preset, as pinned indicators
        self.camera_check = CameraCheck(self.settings.camera.cameras, self.settings.camera.camera_check)

        # RENDER
        self.render = DeepFlowRender(self.board, self.settings.render, self.settings.camera.cameras,
                                     self.settings.camera.camera_check, num_cams=len(self.cameras), max_players=max_players)
        self.render.add_exit_callback(self.stop)
        self.render.add_update_callback(self.camera_check.update)

        self.render.add_update_callback(self._update_poses)

    def start(self) -> None:
        self.panel_server.start()

        if self.person_detector is not None:
            self.person_detector.start()
        for camera in self.cameras:
            camera.start()

        self.tracker.start()
        self.pose_inference.start()
        self.segmentation_predictor.start()
        self.optical_flow_predictor.start()

        self.osc_sound.start()

        if self.player:
            self.player.start()
        if self.recorder:
            self.recorder.start()

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
        if self.recorder:
            self.recorder.stop()

        self.tracker.stop()
        self.osc_sound.stop()

        self.pose_inference.stop()
        if self.person_detector is not None:
            self.person_detector.stop()
        self.segmentation_predictor.stop()
        self.optical_flow_predictor.stop()

        for camera in self.cameras:
            camera.join(timeout=10)

        self.is_finished = True
