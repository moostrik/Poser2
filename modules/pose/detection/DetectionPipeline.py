# Standard library imports
from dataclasses import dataclass
from threading import Event, Lock, Thread
import time
import traceback

# Third-party imports
import numpy as np

# Pose imports
from modules.pose.detection.Detection import Detection, DetectionInput, DetectionOutput, POSE_MODEL_TYPE_NAMES, POSE_MODEL_WIDTH, POSE_MODEL_HEIGHT
from modules.pose.detection.ImageProcessor import ImageProcessor
from modules.pose.features import AngleFactory, BBoxFeature, Point2DFeature
from modules.pose.Pose import Pose, PoseDict, PoseDictCallback

# Local application imports
from modules.cam.depthcam.Definitions import FrameType
from modules.Settings import Settings
from modules.tracker.Tracklet import Tracklet, Rect, TrackletDict

from modules.utils.HotReloadMethods import HotReloadMethods

@dataclass
class PendingRequest:
    batch_id: int  # For debugging/logging
    time_stamp: float
    tracklets: list[Tracklet]
    crop_rects: list[Rect]
    crop_images: list[np.ndarray]

class DetectionPipeline(Thread):
    def __init__(self, settings: Settings) -> None:
        super().__init__()

        # Thread coordination - align naming with PoseDetection
        self._shutdown_event: Event = Event()
        self._update_event: Event = Event()

        # Input data
        self.input_mutex: Lock = Lock()
        self.input_tracklets: TrackletDict = {}
        self.input_frames: dict[int, np.ndarray] = {}

        self.batch_counter: int = 0
        self._pending_lock: Lock = Lock()
        self._pending_requests: dict[int, PendingRequest] = {}

        # Configuration
        self.pose_active: bool = settings.pose_active
        self.pose_detector_frame_width: int = POSE_MODEL_WIDTH
        self.pose_detector_frame_height: int = POSE_MODEL_HEIGHT
        self.pose_crop_expansion: float = settings.pose_crop_expansion
        self.max_detectors: int = settings.num_players
        self.verbose: bool = settings.pose_verbose

        # Components
        self.pose_detector: Detection | None = None
        self.image_processor: ImageProcessor = ImageProcessor(self.pose_crop_expansion, self.pose_detector_frame_width, self.pose_detector_frame_height)

        if self.pose_active:
            self.pose_detector = Detection(
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

        # Start detector first
        if self.pose_detector is not None:
            self.pose_detector.register_callback(self._notify_detection_callback)
            self.pose_detector.start()

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

        # Stop detector
        if self.pose_detector is not None:
            self.pose_detector.stop()

    def run(self) -> None:
        """Main pipeline loop - processes tracklets and submits to detector"""
        while not self._shutdown_event.is_set():
            # Wait for update signal
            self._update_event.wait()

            # Check shutdown again after waking
            if self._shutdown_event.is_set():
                break

            self._update_event.clear()

            try:
                self._process()

            except Exception as e:
                if self.verbose:
                    print(f"PosePipeline Error: {str(e)}")
                    traceback.print_exc()

    def _process(self) -> None:
        tracklets: list[Tracklet] = list(self.get_tracklets().values())

        if not tracklets:
            # print("No tracklets to process")
            return

        self.batch_counter += 1
        batch_id: int = self.batch_counter
        time_stamp: float = time.time()
        pose_images: list[np.ndarray] = []
        pose_crop_rects: list[Rect] = []

        for tracklet in tracklets:
            cam_image: np.ndarray = self._get_image(tracklet.cam_id)
            pose_image, pose_crop_rect = self.image_processor.process_pose_image(tracklet, cam_image)
            pose_images.append(pose_image)
            pose_crop_rects.append(pose_crop_rect)

        pending_request = PendingRequest(
            batch_id=batch_id,
            time_stamp=time_stamp,
            tracklets=tracklets,
            crop_rects=pose_crop_rects,
            crop_images=pose_images
        )

        with self._pending_lock:
            self._pending_requests[batch_id] = pending_request

        if self.pose_detector is not None and self.pose_detector.is_ready:
            pose_data_input = DetectionInput(pending_request.batch_id, pending_request.crop_images)
            self.pose_detector.submit_batch(pose_data_input)
        else:
            print("PosePipeline: Pose detector not ready, skipping submission")

    def _notify_detection_callback(self, batch_detection: DetectionOutput) -> None:

        """Handle completed pose detection results"""
        with self._pending_lock:
            pending_request: PendingRequest | None = self._pending_requests.pop(batch_detection.batch_id, None)
            # delete andy pending request with a lower batch id to avoid memory leak
            obsolete_keys: list[int] = [key for key in self._pending_requests if key < batch_detection.batch_id]
            # if self.verbose and obsolete_keys:
            #     print(f"PosePipeline: Cleaning up {len(obsolete_keys)} obsolete pending requests: {obsolete_keys}")
            for key in obsolete_keys:
                del self._pending_requests[key]

        if pending_request is None:
            if self.verbose:
                print(f"PosePipeline: No pending request found for batch ID {batch_detection.batch_id}")
            return

        pose_dict: PoseDict = {}

        for i, tracklet in enumerate(pending_request.tracklets):
            pose = Pose(
                track_id=tracklet.id,
                tracklet=tracklet,
                bbox = BBoxFeature.from_rect(pending_request.crop_rects[i]),
                time_stamp = pending_request.time_stamp,
                lost=tracklet.is_removed,
                points = Point2DFeature(batch_detection.point_batch[i], scores=batch_detection.score_batch[i])
            )

            object.__setattr__(pose, "crop_image", pending_request.crop_images[i])
            pose_dict[tracklet.id] = pose

        self._notify_pose_callbacks(pose_dict)

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

    def _get_image(self, id: int) -> np.ndarray:
        """Get the camera image for a specific camera ID (thread-safe)"""
        with self.input_mutex:
            image: np.ndarray | None = self.input_frames.get(id)
            if image is None:
                raise ValueError(f"PosePipeline: No image available for camera ID {id}")
            return image

    def notify_update(self) -> None:
        """Signal the pipeline to process current tracklets"""
        if not self._shutdown_event.is_set():
            self._update_event.set()

    # CALLBACK METHODS
    def add_callback(self, callback: PoseDictCallback) -> None:
        """Register a callback to be invoked when pose detection completes"""

        with self.callback_lock:
            self.pose_output_callbacks.add(callback)

    def _notify_pose_callbacks(self, poses: PoseDict) -> None:
        """Invoke all registered pose output callbacks"""
        with self.callback_lock:
            for callback in self.pose_output_callbacks:
                try:
                    callback(poses)
                except Exception as e:
                    print(f"PosePipeline: Error in pose output callback: {str(e)}")
                    traceback.print_exc()