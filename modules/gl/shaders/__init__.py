"""OpenGL shaders useful to any app, grouped by kind in subpackages; every one is used as `shaders.<Operation>`."""

from .draw import Blit, DrawRoi, FillCircles, FillRect, StrokeRect
from .color import Contrast, Exposure, Hsl, Hsv, Lut
from .filter import Blur, Sharpen, EdgeSketch, MaskDilate, MaskErode
from .mask import MaskApply, MaskHardStep, DetailThreshold, Tint
from .composite import Blend
from .transition import BarnDoorWipe, NoiseBlend, NoiseSimplexBlend
from .generate import Noise, NoiseSimplex
from .video import Nv12, Nv12Plane, bt709_video_range
from .depth_align import DepthAlign
