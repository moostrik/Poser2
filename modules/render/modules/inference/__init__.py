"""The layers that draw the inference's output: the crops, the segmentation, its foreground colorized and the capture."""

from .crop_layer import                    CropLayer, CropLayerSettings
from .crop_image_layer import              CropImageLayer
from .segmentation_mask_layer import       SegmentationMaskLayer
from .segmentation_foreground_layer import SegmentationForegroundLayer
from .colorize_layer import                Colorize, ColorizeLayer, ColorizeSettings
from .capture_image_layer import           CaptureImageLayer
from .capture_depth_layer import           CaptureDepthLayer
