"""The view stage: the flow's view of the capture, its window on the camera image and the depth read into it."""

from .view_window import DepthRead, fixed_window, texture_roi, view_span, depth_step, view_per_depth_pixel, place_points, edge_aware_depth, mask_sided_depth, joint_bilateral_depth
from .shaders import DepthUpscale
