"""Shaders that filter an image by its neighborhood: blur, sharpen, edges, morphology."""

from .blur import Blur, KERNEL_SIGMA
from .sharpen import Sharpen
from .edge_sketch import EdgeSketch
from .mask_dilate import MaskDilate
from .mask_erode import MaskErode
