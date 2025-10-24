# Standard library imports
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Optional

# Third-party imports
import numpy as np

# Local application imports
from modules.cam.depthcam.Definitions import FrameType
from modules.tracker.Tracklet import Tracklet, Rect, TrackletDict
from modules.pose.Pose import Pose, PoseDict, PoseDictCallback
from modules.pose.PoseDetection import PoseDetection, POSE_MODEL_TYPE_NAMES
from modules.pose.PoseImageProcessor import PoseImageProcessor
from modules.Settings import Settings

from modules.utils.HotReloadMethods import HotReloadMethods


class PosePipeline(Thread):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._stop_event = Event()
        self.tracklet_input_queue: Queue[TrackletDict] = Queue()

        self.input_mutex: Lock = Lock()
        self.input_frames: dict[int, np.ndarray] = {}

        self.pose_active: bool = settings.pose_active
        self.pose_detector_frame_width: int = 192
        self.pose_detector_frame_height: int = 256
        self.pose_crop_expansion: float = settings.pose_crop_expansion
        self.max_detectors: int = settings.num_players
        self.pose_detector: PoseDetection | None = None
        self.image_processor: PoseImageProcessor = PoseImageProcessor(
            crop_expansion=self.pose_crop_expansion,
            output_width=self.pose_detector_frame_width,
            output_height=self.pose_detector_frame_height
        )


        if self.pose_active:
            self.pose_detector = PoseDetection(settings.path_model, settings.pose_model_type, settings.pose_model_warmups, settings.camera_fps, settings.pose_conf_threshold, settings.pose_verbose)
            print('Pose Detection:', 'model', POSE_MODEL_TYPE_NAMES[settings.pose_model_type.value])
        else:
            print('Pose Detection: Disabled')

        # Callbacks
        self.callback_lock = Lock()
        self.pose_output_callbacks: set[PoseDictCallback] = set()

        hot_reloader = HotReloadMethods(self.__class__)

    def start(self) -> None:
        if super().is_alive():
            print("PosePipeline is already running.")
            return

        # Start detectors
        if self.pose_detector is not None:
            self.pose_detector.addMessageCallback(self._notify_pose_callback)
            self.pose_detector.start()

        super().start()

    def stop(self) -> None:
        self._stop_event.set()
        self.join()
        with self.callback_lock:
            self.pose_output_callbacks.clear()

        if self.pose_detector is not None:
            self.pose_detector.stop()

    def run(self) -> None:
        """
        Update the pose pipeline with a new tracklet.
        This method is called when a new tracklet is detected or updated.
        """
        while not self._stop_event.is_set():
            try:
                tracklets: TrackletDict = self.tracklet_input_queue.get(block=True, timeout=0.01)
                pre_poses: PoseDict = self._process_tracklets(tracklets)
                if self.pose_detector is not None:
                    self.pose_detector.add_poses(pre_poses)
                    self.pose_detector.notify_update()
            except Empty:
                continue

    def _process_tracklets(self, tracklets: TrackletDict) -> PoseDict:
        poses: PoseDict = {}
        for tracklet in tracklets.values():
            pose: Pose = self._process_tracklet(tracklet)
            poses[tracklet.id] = pose
        return poses

    def _process_tracklet(self, tracklet: Tracklet) -> Pose:
        pose_image: Optional[np.ndarray] = None
        pose_crop_rect: Optional[Rect] = None
        cam_image: Optional[np.ndarray] = self._get_image(tracklet.cam_id)

        if cam_image is not None and tracklet.is_active:
            pose_image, pose_crop_rect = self.image_processor.process_pose_image(tracklet, cam_image)
            # if tracklet.is_lost:
            #     print("lost", tracklet.id, pose_crop_rect)

        pose = Pose(
            tracklet=tracklet,
            crop_rect = pose_crop_rect,
            crop_image = pose_image
        )

        return pose

     # INPUTS
    def add_tracklets(self, tracklets: TrackletDict) -> None:
        self.tracklet_input_queue.put(tracklets)

    def set_image(self, id: int, frame_type: FrameType, image: np.ndarray) -> None :
        if frame_type != FrameType.VIDEO:
            return
        with self.input_mutex:
            self.input_frames[id] = image

    def _get_image(self, id: int) -> Optional[np.ndarray]:
        with self.input_mutex:
            if not self.input_frames.get(id) is None:
                return self.input_frames[id]
            return None

    # External Output Callbacks
    def add_pose_callback(self, callback: PoseDictCallback) -> None:
        with self.callback_lock:
            self.pose_output_callbacks.add(callback)

    def _notify_pose_callback(self, poses: PoseDict) -> None:
        with self.callback_lock:
            for callback in self.pose_output_callbacks:
                callback(poses)