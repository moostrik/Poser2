# Standard library imports
from queue import Empty
from threading import Event, Lock, Thread
from typing import Optional
import traceback

# Third-party imports
import numpy as np
from pandas import Timestamp

# Local application imports
from modules.cam.depthcam.Definitions import FrameType
from modules.tracker.Tracklet import Tracklet, Rect, TrackletDict
from modules.pose.Pose import Pose, PoseDict, PoseDictCallback
from modules.pose.PoseDetection import PoseDetection, POSE_MODEL_TYPE_NAMES, POSE_MODEL_WIDTH, POSE_MODEL_HEIGHT
from modules.pose.PoseImageProcessor import PoseImageProcessor
from modules.Settings import Settings

from modules.utils.HotReloadMethods import HotReloadMethods


class PosePipeline(Thread):
    def __init__(self, settings: Settings) -> None:
        super().__init__()

        # Thread coordination - align naming with PoseDetection
        self._shutdown_event: Event = Event()
        self._update_event: Event = Event()

        # Input data
        self.input_mutex: Lock = Lock()
        self.input_tracklets: TrackletDict = {}
        self.input_frames: dict[int, np.ndarray] = {}

        # Configuration
        self.pose_active: bool = settings.pose_active
        self.pose_detector_frame_width: int = POSE_MODEL_WIDTH
        self.pose_detector_frame_height: int = POSE_MODEL_HEIGHT
        self.pose_crop_expansion: float = settings.pose_crop_expansion
        self.max_detectors: int = settings.num_players
        self.verbose: bool = settings.pose_verbose

        # Components
        self.pose_detector: PoseDetection | None = None
        self.image_processor: PoseImageProcessor = PoseImageProcessor(
            crop_expansion=self.pose_crop_expansion,
            output_width=self.pose_detector_frame_width,
            output_height=self.pose_detector_frame_height
        )

        if self.pose_active:
            self.pose_detector = PoseDetection(
                settings.path_model,
                settings.pose_model_type,
                settings.pose_model_warmups,
                settings.pose_conf_threshold,
                settings.pose_verbose
            )
            print(f'Pose Detection: model={POSE_MODEL_TYPE_NAMES[settings.pose_model_type.value]}')
        else:
            print('Pose Detection: Disabled')

        # Callbacks
        self.callback_lock: Lock = Lock()
        self.pose_output_callbacks: set[PoseDictCallback] = set()

        self._hot_reloader = HotReloadMethods(self.__class__)

    @property
    def is_running(self) -> bool:
        """Check if the pipeline is running"""
        return not self._shutdown_event.is_set() and self.is_alive()

    def start(self) -> None:
        """Start the pose pipeline and detector"""
        if self.is_alive():
            print("PosePipeline is already running.")
            return

        # Start detector first
        if self.pose_detector is not None:
            self.pose_detector.add_poses_callback(self._notify_pose_callback)
            self.pose_detector.start()

            # Wait for detector to be ready (optional, but safer)
            import time
            timeout = 10.0  # seconds
            start_time = time.time()
            while not self.pose_detector.is_ready:
                if time.time() - start_time > timeout:
                    print("Warning: PoseDetection not ready after timeout")
                    break
                time.sleep(0.1)

            if self.verbose and self.pose_detector.is_ready:
                print("PosePipeline: PoseDetection ready")

        super().start()

    def stop(self) -> None:
        """Stop the pose pipeline gracefully"""
        # Signal shutdown
        self._shutdown_event.set()
        self._update_event.set()

        # Wait for pipeline thread to stop
        self.join(timeout=2.0)
        if self.is_alive() and self.verbose:
            print("Warning: PosePipeline thread did not stop cleanly")

        # Clear callbacks
        with self.callback_lock:
            self.pose_output_callbacks.clear()

        # Stop detector
        if self.pose_detector is not None:
            self.pose_detector.stop()

    def run(self) -> None:
        """Main pipeline loop - processes tracklets and submits to detector"""
        while not self._shutdown_event.is_set():
            # Wait for update signal
            self._update_event.wait(timeout=1.0)

            # Check shutdown again after waking
            if self._shutdown_event.is_set():
                break

            self._update_event.clear()

            try:
                # Get current tracklets
                tracklets: TrackletDict = self.get_tracklets()

                # Process tracklets into pre-poses
                pre_poses: PoseDict = self._process_tracklets(tracklets)

                # Submit to detector if ready
                if self.pose_detector is not None and self.pose_detector.is_ready:
                    self.pose_detector.submit_poses(pre_poses)
                elif self.pose_detector is not None and self.verbose:
                    print("PosePipeline: Detector not ready, skipping submission")

            except Exception as e:
                if self.verbose:
                    print(f"PosePipeline Error: {str(e)}")
                    traceback.print_exc()

        if self.verbose:
            print("PosePipeline: Pipeline thread stopped")

    def _process_tracklets(self, tracklets: TrackletDict) -> PoseDict:
        """Convert tracklets to pre-poses (with crop images but no keypoints)"""
        poses: PoseDict = {}
        time_stamp: Timestamp = Timestamp.now()

        for tracklet in tracklets.values():
            pose: Pose = self._process_tracklet(tracklet, time_stamp)
            poses[tracklet.id] = pose

        # if self.verbose:
        #     print(f"PosePipeline In: {[(pid, f'{pose.time_stamp.second}.{int(pose.time_stamp.microsecond/1000):03d}') for pid, pose in poses.items()]}")

        return poses

    def _process_tracklet(self, tracklet: Tracklet, time_stamp: Timestamp) -> Pose:
        """Process a single tracklet into a pre-pose"""
        pose_image: Optional[np.ndarray] = None
        pose_crop_rect: Optional[Rect] = None
        cam_image: Optional[np.ndarray] = self._get_image(tracklet.cam_id)

        if cam_image is not None and tracklet.is_being_tracked:
            pose_image, pose_crop_rect = self.image_processor.process_pose_image(tracklet, cam_image)

        pose = Pose(
            tracklet=tracklet,
            crop_rect=pose_crop_rect,
            crop_image=pose_image,
            time_stamp=time_stamp
        )

        return pose

    # INPUT METHODS
    def set_tracklets(self, tracklets: TrackletDict) -> None:
        """Update the current tracklets (thread-safe)"""
        with self.input_mutex:
            self.input_tracklets = tracklets

    def get_tracklets(self) -> TrackletDict:
        """Get the current tracklets (thread-safe)"""
        with self.input_mutex:
            return self.input_tracklets

    def set_image(self, id: int, frame_type: FrameType, image: np.ndarray) -> None:
        """Update the camera image for a specific camera ID"""
        if frame_type != FrameType.VIDEO:
            return
        with self.input_mutex:
            self.input_frames[id] = image

    def _get_image(self, id: int) -> Optional[np.ndarray]:
        """Get the camera image for a specific camera ID (thread-safe)"""
        with self.input_mutex:
            return self.input_frames.get(id)

    def notify_update(self) -> None:
        """Signal the pipeline to process current tracklets"""
        if not self._shutdown_event.is_set():
            self._update_event.set()

    # CALLBACK METHODS
    def add_pose_callback(self, callback: PoseDictCallback) -> None:
        """Register a callback to be invoked when pose detection completes"""

        with self.callback_lock:
            self.pose_output_callbacks.add(callback)

    def _notify_pose_callback(self, poses: PoseDict) -> None:
        """Internal callback invoked by PoseDetection when poses are ready

        Note: Error handling is performed by PoseDetection's callback worker thread.
              Any exceptions raised here will be caught and logged by PoseDetection.
        """
        # if self.verbose:
        #     print(f"PosePipeline Out: {[(pid, f'{pose.time_stamp.second}.{int(pose.time_stamp.microsecond/1000):03d}') for pid, pose in poses.items()]}")

        with self.callback_lock:
            for callback in self.pose_output_callbacks:
                callback(poses)