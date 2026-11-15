"""OpenGL shaders useful to any app, grouped by kind in subpackages; every one is used as `shaders.<Operation>`."""

from .draw import Blit, FillCircles, FillRect, StrokeRect
from .color import BrightnessContrast, Exposure, Hsl, Hsv, Lut
from .filter import Blur, Sharpen, Sobel, Dilate, Erode
from .mask import MaskApply, Threshold, AdaptiveThreshold, ColorFill
from .composite import Mix
from .transition import BarnDoorWipe, GradientWipe
from .generate import WhiteNoise, SimplexNoise
from .video import Nv12, Nv12Plane, bt709_video_range
from .depth_align import DepthAlign
