"""Shaders that filter an image by its neighborhood: blur, sharpen, edges, morphology, edge-keeping smoothing."""

from .blur import Blur, KERNEL_SIGMA
from .sharpen import Sharpen
from .sobel import Sobel
from .dilate import Dilate
from .erode import Erode
from .guided_filter import GuidedFilter
from .mosaic import Mosaic
from .displace import Displace
from .median import Median
from .bilateral import Bilateral
from .dual_filter import DualFilter
from .bloom import Bloom
