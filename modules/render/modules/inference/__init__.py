"""The layers that draw the inference's output: the crops, the segmentation and the inference frame."""

from .crop_layer import                    CropLayer
from .crop_image_layer import              CropImageLayer
from .segmentation_mask_layer import       SegmentationMaskLayer
from .segmentation_foreground_layer import SegmentationForegroundLayer
from .inference_frame_image_layer import   InferenceFrameImageLayer
from .inference_frame_depth_layer import   InferenceFrameDepthLayer
