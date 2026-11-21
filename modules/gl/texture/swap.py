"""Swap: a ping-pong pair, `current` written now and `back` the previous one; TextureSwap: a pair of one kind of
texture, used as a texture it is always its current buffer. `back` is read inside the pass that writes the current,
never kept."""

from typing import Generic, Protocol, TypeVar

from OpenGL.GL import Constant

from .texture_base import TextureBase


class Buffer(Protocol):
    """What a pair holds: a GL object that is allocated and freed."""
    @property
    def allocated(self) -> bool: ...
    def deallocate(self) -> None: ...


B = TypeVar('B', bound=Buffer)
T = TypeVar('T', bound=TextureBase)


class Swap(Generic[B]):
    """Two buffers of one kind, `current` and `back`, that trade places at `swap`. A named pair allocates them."""

    def __init__(self, first: B, second: B) -> None:
        self._buffers: tuple[B, B] = (first, second)
        self._swap_state: int = 0

    @property
    def current(self) -> B:
        """The buffer written now."""
        return self._buffers[self._swap_state]

    @property
    def back(self) -> B:
        """The other buffer, the previous one, to read while the current is written; it is the current one after the
        next swap."""
        return self._buffers[1 - self._swap_state]

    @property
    def allocated(self) -> bool:
        return self._buffers[0].allocated and self._buffers[1].allocated

    def swap(self) -> None:
        """The back buffer becomes the current one."""
        self._swap_state = 1 - self._swap_state

    def deallocate(self) -> None:
        for buffer in self._buffers:
            buffer.deallocate()


class TextureSwap(Swap[T]):
    """A pair of one kind of texture; what a texture has, read from the current one. A named pair (`SwapFbo`,
    `SwapTexture3D`, `SwapTexture2DArray`) is also of its buffers' class and allocates them."""

    @property
    def width(self) -> int:
        return self.current.width

    @property
    def height(self) -> int:
        return self.current.height

    @property
    def internal_format(self) -> Constant | int:
        return self.current.internal_format

    @property
    def format(self) -> Constant:
        return self.current.format

    @property
    def data_type(self) -> int:
        return self.current.data_type

    @property
    def tex_id(self) -> int:
        return self.current.tex_id

    def bind(self, unit: int) -> None:
        self.current.bind(unit)

    def clear(self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 0.0) -> None:
        """The current buffer cleared."""
        self.current.clear(r, g, b, a)

    def clear_back(self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 0.0) -> None:
        """The back buffer cleared."""
        self.back.clear(r, g, b, a)

    def clear_all(self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 0.0) -> None:
        """Both buffers cleared."""
        for buffer in self._buffers:
            buffer.clear(r, g, b, a)

    def set_wrap(self, wrap: int, border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        """The x and y wrap of both buffers changed (GL thread only)."""
        for buffer in self._buffers:
            buffer.set_wrap(wrap, border_color)
