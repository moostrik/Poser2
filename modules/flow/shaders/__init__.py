"""Shared flow shaders — used across bridge, fluid, and visualization layers."""

from .add import Add
from .add_region import AddRegion
from .blit import Blit
from .channel_add import ChannelAdd
from .channel_add_region import ChannelAddRegion
from .channel_copy import ChannelCopy
from .channel_set_region import ChannelSetRegion
from .clamp import Clamp
from .dampen import Dampen
from .gaussian_blur import GaussianBlur
from .hsv import HSV
from .magnitude_shader import MagnitudeShader
from .normalize import Normalize
from .set import Set
from .set_region import SetRegion
from .trail import Trail