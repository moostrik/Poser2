"""The view stage: the flow's view of the capture, its window on the camera image and the depth read into it."""

from .view_window import fixed_window, view_span, depth_step, view_per_depth_pixel, place_points
from .depth_read import depth_crop, edge_aware_depth, upscaled_depth, push_pull, pyramid_sizes
from .depth_fill import DepthFill
from .shaders import DepthCrop, DepthUpscale
