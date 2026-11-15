"""Shaders that filter an image by its neighborhood: blur, sharpen, edges, morphology."""

from .blur import Blur, KERNEL_SIGMA
from .sharpen import Sharpen
from .sobel import Sobel
from .dilate import Dilate
from .erode import Erode
from .guided_filter import GuidedFilter
