"""The view stage: the flow's view of the capture, its window on the camera image and the depth upscaled into it."""

from .view_window import fixed_window, view_span, depth_step, view_per_depth_pixel, place_points
from .depth_fill import DepthFill, pyramid_sizes
from .shaders import DepthCrop, DepthUpscale
