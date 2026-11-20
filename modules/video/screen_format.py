"""ScreenFormat: the format of a screen recording's frames, its size, its chroma and its bit depth."""

from dataclasses import dataclass
from enum import Enum, IntEnum, auto

import numpy as np


class Chroma(Enum):
    """The color resolution of a screen frame's YUV planes."""
    YUV420 = auto()     # Cb and Cr at half width and half height, interleaved: NV12, or P010 at 10 bits
    YUV444 = auto()     # Cb and Cr at full size: three planes (yuv444p), or at 10 bits packed in one (xv30le)


class BitDepth(IntEnum):
    """The bits of a screen frame's samples."""
    BIT_8 =  8
    BIT_10 = 10


@dataclass(frozen=True)
class ScreenFormat:
    """A screen frame's size (px, even), chroma and bit depth; its planes lie one after the other, top row first, in
    one (rows, width) array of `dtype`. 8 bits: bytes, Y then Cb Cr. 4:2:0 at 10 bits: P010, as NV12 in uint16 with
    the 10 bits high. 4:4:4 at 10 bits: one plane of uint32, xv30le, Cb in bits 0-9, Y in 10-19, Cr in 20-29."""
    width: int
    height: int
    chroma: Chroma = Chroma.YUV444
    bit_depth: BitDepth = BitDepth.BIT_8

    @property
    def packed(self) -> bool:
        """Whether Y, Cb and Cr share one plane (4:4:4 at 10 bits)."""
        return self.chroma is Chroma.YUV444 and self.bit_depth is BitDepth.BIT_10

    @property
    def rows(self) -> int:
        """The array's rows: the Y plane's, then the chroma's (half for 4:2:0, two planes for 4:4:4), or one plane
        when packed."""
        if self.chroma is Chroma.YUV420:
            return self.height * 3 // 2
        return self.height if self.packed else self.height * 3

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.width)

    @property
    def dtype(self) -> type[np.unsignedinteger]:
        if self.bit_depth is BitDepth.BIT_8:
            return np.uint8
        return np.uint32 if self.packed else np.uint16

    @property
    def frame_bytes(self) -> int:
        return self.rows * self.width * np.dtype(self.dtype).itemsize
