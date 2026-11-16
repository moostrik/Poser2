"""The view stage: the flow's view of the capture, its window on the camera image and the depth read into it."""

from .view_window import fixed_window, view_span, depth_step, view_per_depth_pixel, place_points
from .depth_read import depth_cut, edge_aware_depth, mask_sided_depth
from .shaders import DepthCut, DepthUpscale
