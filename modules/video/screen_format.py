"""ScreenFormat: the format of a screen recording's frames, its size and its chroma."""

from dataclasses import dataclass
from enum import Enum, auto


class Chroma(Enum):
    """The color resolution of a screen frame's YUV planes."""
    YUV420 = auto()     # Cb and Cr at half width and half height, interleaved: NV12
    YUV444 = auto()     # Cb and Cr at full size, each its own plane: yuv444p


@dataclass(frozen=True)
class ScreenFormat:
    """A screen frame's size (px, even) and chroma; its planes lie one after the other, top row first, in one
    (rows, width) uint8 array: Y, then Cb Cr."""
    width: int
    height: int
    chroma: Chroma = Chroma.YUV444

    @property
    def rows(self) -> int:
        """The array's rows: the Y plane's, then the chroma's (half for 4:2:0, two planes for 4:4:4)."""
        return self.height * 3 // 2 if self.chroma is Chroma.YUV420 else self.height * 3

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.width)

    @property
    def frame_bytes(self) -> int:
        return self.rows * self.width
