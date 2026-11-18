"""OpenGL shaders useful to any app, grouped by kind in subpackages; every one is used as `shaders.<Operation>`."""

from .draw import Blit, FillCircles, FillRect, StrokeRect
from .color import BrightnessContrast, Exposure, Hsl, Hsv, Lut, Colorize, ToneMap, ToneMapOperator, Levels, \
    ColorMatrix, saturation_matrix, Posterize, Dither, DitherPattern, Vignette, Grade
from .filter import Blur, Sharpen, Sobel, Dilate, Erode, GuidedFilter, Mosaic, Displace, Median, Bilateral, DualFilter
from .mask import MaskApply, Threshold, AdaptiveThreshold, ColorFill, MaskMultiply, MaskAntialias, ChromaKey, LumaKey, \
    LumaMatte, JumpFlood, Feather
from .composite import Mix, WeightedSum, Blend, BlendShaderMode
from .transition import BarnDoorWipe, GradientWipe
from .generate import WhiteNoise, SimplexNoise
from .video import Nv12, Nv12Plane, bt709_video_range
