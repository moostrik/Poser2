"""The texture formats: for each GL internal format, the pixel format, the types an upload is given and its channels."""

import logging
from dataclasses import dataclass
from typing import cast

import numpy as np
from OpenGL.GL import (Constant, GL_BGR, GL_BGRA, GL_FLOAT, GL_HALF_FLOAT, GL_NONE, GL_R16, GL_R16F, GL_R32F, GL_R32UI,
                       GL_R8, GL_RED, GL_RED_INTEGER, GL_RG, GL_RG16, GL_RG16F, GL_RG32F, GL_RG8, GL_RGB, GL_RGB10_A2,
                       GL_RGB16F, GL_RGB32F, GL_RGB8, GL_RGBA, GL_RGBA16F, GL_RGBA32F, GL_RGBA8, GL_UNSIGNED_BYTE,
                       GL_UNSIGNED_INT, GL_UNSIGNED_INT_2_10_10_10_REV, GL_UNSIGNED_SHORT)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextureFormat:
    """What GL needs to fill a texture of one internal format. A 16-bit float texture is filled from float32 by a CPU
    upload (`data_type`, GL converts) and from half floats by a CUDA upload (`tensor_type`)."""
    format: Constant            # the pixel format, channels in RGB order
    bgr_format: Constant        # the same in BGR order, an OpenCV image's; one or two channels are the same
    data_type: int              # the type a CPU upload, a read and a clear are given
    tensor_type: int            # the type a CUDA upload is given
    channels: int
    bytes_per_channel: int      # of `tensor_type`


_U8, _U16, _U32, _F16, _F32 = int(GL_UNSIGNED_BYTE), int(GL_UNSIGNED_SHORT), int(GL_UNSIGNED_INT), int(GL_HALF_FLOAT), int(GL_FLOAT)
_RGB10_A2 = int(GL_UNSIGNED_INT_2_10_10_10_REV)

FORMATS: dict[int, TextureFormat] = {int(internal): TextureFormat(*row) for internal, row in (
    (GL_RGB,      (GL_RGB,  GL_BGR,  _U8,  _U8,  3, 1)),
    (GL_RGB8,     (GL_RGB,  GL_BGR,  _U8,  _U8,  3, 1)),
    (GL_RGB16F,   (GL_RGB,  GL_BGR,  _F32, _F16, 3, 2)),
    (GL_RGB32F,   (GL_RGB,  GL_BGR,  _F32, _F32, 3, 4)),
    (GL_RGBA,     (GL_RGBA, GL_BGRA, _U8,  _U8,  4, 1)),
    (GL_RGBA8,    (GL_RGBA, GL_BGRA, _U8,  _U8,  4, 1)),
    (GL_RGB10_A2, (GL_RGBA, GL_BGRA, _RGB10_A2, _RGB10_A2, 4, 1)),
    (GL_RGBA16F,  (GL_RGBA, GL_BGRA, _F32, _F16, 4, 2)),
    (GL_RGBA32F,  (GL_RGBA, GL_BGRA, _F32, _F32, 4, 4)),
    (GL_R8,       (GL_RED,  GL_RED,  _U8,  _U8,  1, 1)),
    (GL_R16,      (GL_RED,  GL_RED,  _U16, _U16, 1, 2)),
    (GL_R16F,     (GL_RED,  GL_RED,  _F32, _F16, 1, 2)),
    (GL_R32F,     (GL_RED,  GL_RED,  _F32, _F32, 1, 4)),
    (GL_R32UI,    (GL_RED_INTEGER, GL_RED_INTEGER, _U32, _U32, 1, 4)),
    (GL_RG8,      (GL_RG,   GL_RG,   _U8,  _U8,  2, 1)),
    (GL_RG16,     (GL_RG,   GL_RG,   _U16, _U16, 2, 2)),
    (GL_RG16F,    (GL_RG,   GL_RG,   _F32, _F16, 2, 2)),
    (GL_RG32F,    (GL_RG,   GL_RG,   _F32, _F32, 2, 4)),
)}


ARRAY_FORMATS: dict[tuple[np.dtype, int], Constant] = {     # an array's element type and channels -> its texture
    (np.dtype(np.uint8), 1): GL_R8,     (np.dtype(np.uint8), 2): GL_RG8,
    (np.dtype(np.uint8), 3): GL_RGB8,   (np.dtype(np.uint8), 4): GL_RGBA8,
    (np.dtype(np.float16), 1): GL_R16F, (np.dtype(np.float16), 2): GL_RG16F,
    (np.dtype(np.float16), 3): GL_RGB16F, (np.dtype(np.float16), 4): GL_RGBA16F,
    (np.dtype(np.float32), 1): GL_R32F, (np.dtype(np.float32), 2): GL_RG32F,
    (np.dtype(np.float32), 3): GL_RGB32F, (np.dtype(np.float32), 4): GL_RGBA32F,
}


def array_format(dtype: np.dtype | type, channels: int) -> Constant:
    """The internal format of a texture for an array of `dtype` with `channels` (1 for a 2D array); GL_NONE for one
    the table does not hold."""
    internal_format: Constant | None = ARRAY_FORMATS.get((np.dtype(dtype), channels))
    return GL_NONE if internal_format is None else internal_format


def texture_format(internal_format: Constant | int) -> TextureFormat | None:
    """The row of an internal format; None, with a warning, for one the table does not hold."""
    row: TextureFormat | None = FORMATS.get(cast(int, internal_format))     # a GL constant is an int
    if row is None:
        logger.warning("internal format %s not supported", internal_format)
    return row


def pixel_format(internal_format: Constant | int, bgr: bool = False) -> Constant:
    """The pixel format of an upload or a read, in RGB order or BGR order; GL_NONE for an unsupported format."""
    row = texture_format(internal_format)
    return GL_NONE if row is None else (row.bgr_format if bgr else row.format)


def data_type(internal_format: Constant | int) -> int:
    """The type a CPU upload, a read and a clear are given; GL_NONE for an unsupported format."""
    row = texture_format(internal_format)
    return int(GL_NONE) if row is None else row.data_type


def tensor_type(internal_format: Constant | int) -> int:
    """The type a CUDA upload is given; GL_FLOAT for an unsupported format."""
    row = texture_format(internal_format)
    return _F32 if row is None else row.tensor_type


def channel_count(internal_format: Constant | int) -> int:
    """The channels of the format; 1 for an unsupported format."""
    row = texture_format(internal_format)
    return 1 if row is None else row.channels


def bytes_per_channel(internal_format: Constant | int) -> int:
    """The bytes of one channel of a CUDA upload; 4 for an unsupported format."""
    row = texture_format(internal_format)
    return 4 if row is None else row.bytes_per_channel
