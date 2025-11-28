
from threading import Lock

import numpy as np

from ...nodes import ImageCropProcessor, ImageCropProcessorConfig, SimilarityExtractor, SimilarityExtractorConfig
from ..ProcessorTracker import ProcessorTracker, TOutput_Callback
from ...similarity import SimilarityBatch

from modules.cam.depthcam.Definitions import FrameType

class ImageCropProcessorTracker(ProcessorTracker[np.ndarray, np.ndarray]):
    """Convenience tracker for cropping images based on poses."""

    def __init__(self, num_tracks: int, config: ImageCropProcessorConfig) -> None:
        """
        Args:
            num_tracks: Number of pose/image slots to track.
            config: Configuration for the ImageCropProcessor.
        """
        self._image_lock = Lock()
        self._images: dict[int, np.ndarray] = dict[int, np.ndarray]()

        super().__init__(num_tracks=num_tracks, processor_factory=lambda: ImageCropProcessor(config)
        )

    def set_images(self, image_dict: dict[int, np.ndarray]) -> None:
        """Set images for cropping."""
        self.set(image_dict)

    # convenience method for setting a single image
    # replace by separate module later?
    def set_image(self, id: int, frame_type: FrameType, image: np.ndarray) -> None:
        """Update the camera image for a specific camera ID"""
        if frame_type != FrameType.VIDEO:
            return
        with self._image_lock:
            self._images[id] = image
        self.set(self._images)

    def add_image_callback(self, callback: TOutput_Callback) -> None:
        self.add_output_callback(callback)

    def remove_image_callback(self, callback: TOutput_Callback) -> None:
        self.remove_output_callback(callback)

