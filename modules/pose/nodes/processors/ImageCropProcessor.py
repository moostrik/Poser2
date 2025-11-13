import numpy as np

from modules.pose.nodes._utils.ImageProcessor import ImageProcessor
from modules.pose.nodes.Nodes import ProcessorNode, NodeConfigBase
from modules.pose.Pose import Pose

class ImageCropProcessorConfig(NodeConfigBase):
    """Configuration for pose chase interpolation with automatic change notification."""

    def __init__(self, crop_expansion: float = 0.1, output_width: int = 192, output_height: int = 256) -> None:
        super().__init__()
        self.crop_expansion: float = crop_expansion
        self.output_width: int = output_width
        self.output_height: int = output_height

class ImageCropProcessor(ProcessorNode[np.ndarray, np.ndarray]):
    """Processes poses with a stored camera image to produce cropped images."""

    def __init__(self, config: ImageCropProcessorConfig) -> None:
        self._config: ImageCropProcessorConfig = config
        self._image_processor: ImageProcessor = ImageProcessor(
            crop_expansion=self._config.crop_expansion,
            output_width=self._config.output_width,
            output_height=self._config.output_height
        )
        self._image: np.ndarray | None = None

    def set(self, input_data: np.ndarray) -> None:
        """Set the full camera image."""
        self._image = input_data

    def process(self, pose: Pose) -> np.ndarray:
        """Crop image based on pose bbox."""

        if self._image is None:
            raise RuntimeError("ImageCropProcessor.process called before image was set.")

        result = self._image_processor.process_pose_image(pose.bbox, self._image)

        return result

    def is_ready(self) -> bool:
        return  self._image is not None

    def reset(self) -> None:
        self._image = None