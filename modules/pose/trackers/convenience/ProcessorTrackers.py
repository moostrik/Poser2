import numpy as np
from modules.pose.nodes.processor.ImageCropProcessor import ImageCropProcessor, ImageCropProcessorConfig
from ..ProcessorTracker import ProcessorTracker, Output_Callback

class ImageCropProcessorTracker(ProcessorTracker[np.ndarray, np.ndarray]):
    """Convenience tracker for cropping images based on poses."""

    def __init__(self, num_tracks: int, config: ImageCropProcessorConfig) -> None:
        """
        Args:
            num_tracks: Number of pose/image slots to track.
            config: Configuration for the ImageCropProcessor.
        """

        super().__init__(num_tracks=num_tracks, processor_factory=lambda: ImageCropProcessor(config)
        )

    def add_image_callback(self, callback: Output_Callback) -> None:
        self.add_output_callback(callback)

    def remove_image_callback(self, callback: Output_Callback) -> None:
        self.remove_output_callback(callback)