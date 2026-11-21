"""TextureBase: what a 2D texture, a 3D texture and a 2D array texture share: the GL name, the size and format, the
wrap and filter, bind, clear and read."""

import logging
from typing import ClassVar

import numpy as np
from OpenGL.GL import (Constant, GL_FLOAT, GL_HALF_FLOAT, GL_NONE, GL_ONE, GL_RED, GL_TEXTURE_BORDER_COLOR,
                       GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_SWIZZLE_A, GL_TEXTURE_SWIZZLE_B,
                       GL_TEXTURE_SWIZZLE_G, GL_TEXTURE_SWIZZLE_R, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
                       GL_UNPACK_ALIGNMENT, GL_UNSIGNED_BYTE, GL_UNSIGNED_INT_2_10_10_10_REV, GL_UNSIGNED_SHORT,
                       glBindTexture, glBindTextureUnit, glClearTexImage, glDeleteTextures, glGenTextures,
                       glGetTextureImage, glPixelStorei, glTexParameterfv, glTexParameteri, glTextureParameterfv,
                       glTextureParameteri)

from .texture_format import channel_count, data_type as texture_data_type, pixel_format

logger = logging.getLogger(__name__)

def get_numpy_dtype(data_type: int) -> type | None:
    """The NumPy dtype of an OpenGL data type, or None.

    A 16-bit float texture is read and written as GL_FLOAT, which GL converts to and from half floats, so its array is
    float32 too; only GL_HALF_FLOAT data is float16.
    """
    if data_type == GL_UNSIGNED_BYTE:
        return np.uint8
    if data_type == GL_UNSIGNED_SHORT:
        return np.uint16
    if data_type == GL_UNSIGNED_INT_2_10_10_10_REV:     # one packed RGB10_A2 pixel
        return np.uint32
    if data_type == GL_FLOAT:
        return np.float32
    if data_type == GL_HALF_FLOAT:
        return np.float16
    return None


# The clear colors handed to glClearTexImage, kept for the life of the process: the driver can read a clear
# color after the call returns, and a temporary array's memory is then already reused (measured: a clear
# to 0 left 1.75, 3.75 and the bit pattern of 100 at startup). Zero passes no memory at all.
_CLEAR_VALUES: dict[tuple, np.ndarray] = {}


def clear_value(data_type: int, channels: int, color: tuple[float, float, float, float]) -> np.ndarray | None:
    """The clear color for glClearTexImage: None for zero (GL fills zeros without reading memory), else
    an array that is never freed, one per value."""
    values = color[:channels]
    if all(v == 0.0 for v in values):
        return None
    key = (int(data_type), values)
    if key not in _CLEAR_VALUES:
        if data_type == GL_UNSIGNED_BYTE:
            _CLEAR_VALUES[key] = np.array([v * 255 for v in values], dtype=np.uint8)
        elif data_type == GL_UNSIGNED_SHORT:
            _CLEAR_VALUES[key] = np.array([v * 65535 for v in values], dtype=np.uint16)
        elif data_type == GL_HALF_FLOAT:
            _CLEAR_VALUES[key] = np.array(values, dtype=np.float16)
        else:
            _CLEAR_VALUES[key] = np.array(values, dtype=np.float32)
    return _CLEAR_VALUES[key]


class TextureBase:
    """A GL texture of one target: its name, size and format, the wrap and filter it was made with."""

    TARGET: ClassVar[int]                   # GL_TEXTURE_2D, GL_TEXTURE_3D, GL_TEXTURE_2D_ARRAY
    WRAP_AXES: ClassVar[tuple[int, ...]]    # the axes it is made wrapping

    def __init__(self, interpolation: int, wrap: int, border_color: tuple[float, float, float, float],
                 bgr: bool = False) -> None:
        """`interpolation`: the min and mag filter; `wrap`: on every axis of WRAP_AXES; `border_color`: with
        GL_CLAMP_TO_BORDER; `bgr`: its uploads in BGR order, an OpenCV image's."""
        self.allocated: bool = False
        self.width: int = 0
        self.height: int = 0
        self.internal_format: Constant | int = GL_NONE
        self.format: Constant = GL_NONE
        self.data_type: int = int(GL_NONE)
        self.tex_id: int = 0
        self._interpolation: int = interpolation
        self._wrap: int = wrap
        self._border_color: tuple[float, float, float, float] = border_color
        self._bgr: bool = bgr

    def _create(self, width: int, height: int, internal_format: Constant | int) -> bool:
        """The texture made and bound, its wrap, filter, border and a red one's grayscale swizzle set; the caller
        allocates its storage and unbinds. False for a format without a data type."""
        data_type: int = texture_data_type(internal_format)
        if data_type == GL_NONE:
            return False
        self.width = width
        self.height = height
        self.internal_format = internal_format
        self.format = pixel_format(internal_format, bgr=self._bgr)
        self.data_type = data_type
        self.tex_id = int(glGenTextures(1))

        glBindTexture(self.TARGET, self.tex_id)
        for axis in self.WRAP_AXES:
            glTexParameteri(self.TARGET, axis, self._wrap)
        glTexParameteri(self.TARGET, GL_TEXTURE_MAG_FILTER, self._interpolation)
        glTexParameteri(self.TARGET, GL_TEXTURE_MIN_FILTER, self._interpolation)
        glTexParameterfv(self.TARGET, GL_TEXTURE_BORDER_COLOR, self._border_color)
        if self.format == GL_RED:       # drawn as grayscale
            glTexParameteri(self.TARGET, GL_TEXTURE_SWIZZLE_R, GL_RED)
            glTexParameteri(self.TARGET, GL_TEXTURE_SWIZZLE_G, GL_RED)
            glTexParameteri(self.TARGET, GL_TEXTURE_SWIZZLE_B, GL_RED)
            glTexParameteri(self.TARGET, GL_TEXTURE_SWIZZLE_A, GL_ONE)
        return True

    def set_wrap(self, wrap: int, border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        """The wrap of the x and y axes (S and T) and the border color changed (GL thread only)."""
        if not self.allocated:
            return
        self._wrap = wrap
        self._border_color = border_color
        glTextureParameteri(self.tex_id, GL_TEXTURE_WRAP_S, wrap)
        glTextureParameteri(self.tex_id, GL_TEXTURE_WRAP_T, wrap)
        glTextureParameterfv(self.tex_id, GL_TEXTURE_BORDER_COLOR, border_color)

    def deallocate(self) -> None:
        if not self.allocated:
            return
        self.allocated = False
        self.width = 0
        self.height = 0
        self.internal_format = GL_NONE
        self.format = GL_NONE
        self.data_type = int(GL_NONE)
        glDeleteTextures(1, [self.tex_id])
        self.tex_id = 0

    def bind(self, unit: int) -> None:
        """Bound to texture `unit` for sampling, its sampler's `layout(binding = unit)` (glBindTextureUnit)."""
        glBindTextureUnit(unit, self.tex_id)

    def clear(self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 0.0) -> None:
        """Every texel set to the color, its channels from 0 to 1 (glClearTexImage, GL 4.4)."""
        if not self.allocated:
            return
        clear_data = clear_value(self.data_type, channel_count(self.internal_format), (r, g, b, a))
        glClearTexImage(self.tex_id, 0, self.format, self.data_type, clear_data)

    def _upload_contiguous(self, array: np.ndarray) -> np.ndarray | None:
        """`array` ready for an upload: contiguous (a strided one copied once), its rows byte-aligned; None when not
        allocated."""
        if not self.allocated:
            return None
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)      # numpy rows are tightly packed; GL's default of 4 shears odd rows
        return np.ascontiguousarray(array)

    def _read(self, shape: tuple[int, ...]) -> np.ndarray | None:
        """Level 0 read back to the CPU, an array of `shape` with the channels last when there is more than one;
        None when not allocated or of a data type NumPy does not have."""
        if not self.allocated:
            return None
        dtype = get_numpy_dtype(self.data_type)
        if dtype is None:
            logger.warning("Unsupported data type %s", self.data_type)
            return None
        channels: int = channel_count(self.internal_format)
        pixels = np.zeros(shape if channels == 1 else (*shape, channels), dtype=dtype)
        glGetTextureImage(self.tex_id, 0, self.format, self.data_type, pixels.nbytes, pixels)
        return pixels
