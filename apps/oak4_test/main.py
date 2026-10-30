"""OAK4 test — one OAK4 camera with the pose model on the device, its people tracked and posed on the PC.

The camera finds the people and their keypoints itself (`model_chain`: one model, or a detector and a pose
model on each person's crop), so the PC runs no crop and no pose model: the tracker follows the camera's
tracklets, `PosesFromTracklets` turns them into poses, and the pose pipeline cleans, smooths and interpolates
them for the window. Runs in `.venv-oak4`, the depthai 3 venv (`scripts\\install.bat oak4`); there is no simulator.
"""

from functools import partial

from modules.camera import CameraCheck, Sync
from modules.camera.oak4 import Camera
from modules.settings import presets
from modules.panel import PanelServer
from modules.tracker import OnePerCamTracker, PosesFromTracklets
from modules.pose import configure_features
from modules.pose.pipeline import Domain, PosePipeline
from modules.inference import Uploader

from .render_board import RenderBoard
from .pose.pipeline import pipeline_spec
from .settings import Settings, Stage
from .render import Oak4TestRender

APP_NAME = 'oak4_test'


class Oak4TestMain:
    def __init__(self, simulation: bool = False, interface: bool = True) -> None:
        if simulation:
            raise RuntimeError('oak4_test has no simulator: it needs the OAK4 and the pose model on it')

        self.is_running: bool = False
        self.is_finished: bool = False

        # SETTINGS
        presets.set_app(APP_NAME)
        self.settings = Settings()
        preset_file = presets.startup_path()
        if not presets.load(self.settings, preset_file):
            raise FileNotFoundError(f"No preset found for '{APP_NAME}' at {preset_file}")
        if not interface:
            self.settings.panel.browser = False  # launcher override; the preset cannot turn it back on
        self.settings.initialize()
        self.panel_server = PanelServer(self.settings, self.settings.panel, on_exit=self.stop)

        max_players: int = self.settings.max_players
        p = self.settings.pose

        # BLACKBOARD
        self.board = RenderBoard()

        # CAMERA — the frames go to the GPU for the window; the tracklets carry the keypoints.
        configure_features(max_players)
        self.camera = Camera(self.settings.camera.cam_0)
        self.uploader = Uploader()
        self.frame_sync_bang = Sync(self.settings.camera.frame_sync, False, 'frame_sync')
        self.tracker = OnePerCamTracker(self.settings.camera.tracker, max_players)
        self.tracklet_sync_bang = Sync(self.settings.camera.tracklet_sync, False, 'tracklet_sync')

        self.camera.add_frame_callback(self.uploader.set_image)
        self.camera.add_frame_callback(self.frame_sync_bang.submit_frame)
        self.camera.add_tracklet_callback(self.tracker.submit_cam_tracklets)
        self.camera.add_tracklet_callback(self.board.set_camera_tracklets)
        self.camera.add_tracklet_callback(self.tracklet_sync_bang.submit_tracklets)

        # POSES — the camera's keypoints in the host crop's convention, which needs the delivered frame's aspect.
        width, height = self.camera.frame_size
        self.poses_from_tracklets = PosesFromTracklets(p.tracklets, max_players, frame_aspect=width / height)
        self.tracker.add_tracklet_callback(self.poses_from_tracklets.set_tracklets)
        self.tracklet_sync_bang.add_sync_callback(self.tracker.update)
        self.frame_sync_bang.add_sync_callback(self._frame_bang)

        # POSE PIPELINE — the stages are declared in pose/pipeline.py. The camera's frame bang drives the frame
        # domain; the render tick drives the output domain (LERP). Every stage's frames and window go to the board.
        self.pipeline = PosePipeline(pipeline_spec(p, self.poses_from_tracklets))
        for stage in Stage:
            self.pipeline.add_frames_callback(stage, partial(self.board.set_frames, stage))
            self.board.set_window_source(stage, self.pipeline.window(stage))

        # CAMERA CHECK — the camera's IMU reading and frame rate against the preset, as pinned indicators
        self.camera_check = CameraCheck(self.settings.camera.cameras, self.settings.camera.camera_check)

        # RENDER
        self.render = Oak4TestRender(self.board, self.settings.render, self.settings.camera.cam_0,
                                     self.settings.camera.camera_check)
        self.render.add_exit_callback(self.stop)
        self.render.add_update_callback(self.camera_check.update)
        self.render.add_update_callback(self._update_poses)

    def start(self) -> None:
        self.panel_server.start()
        self.camera.start()
        self.tracker.start()
        self.is_running = True
        self.render.start()

    def _frame_bang(self, now: float) -> None:
        """The camera's frame bang: the newest image to the board, and the tracklets' poses into the pipeline."""
        images, _ = self.uploader.snapshot()
        self.board.set_camera_images(images)
        self.poses_from_tracklets.process(now)

    def _update_poses(self, now: float) -> None:
        """The render tick drives the pipeline's output domain (LERP) at the frame's time."""
        self.pipeline.update(Domain.OUTPUT, now)

    def stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False

        self.panel_server.stop()
        self.render.stop()
        self.camera.stop()
        self.tracker.stop()
        self.camera.join(timeout=10)

        self.is_finished = True
