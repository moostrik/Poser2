"""The buffers a double buffer hands out: valid until its owner's next swap.

`SwapFbo`, `SwapTexture3D` and `SwapTexture2DArray` hand out their current and back buffer, from
`texture` and `back_texture`, as a handle of the buffer's own type. It reads as that buffer; after
the owner swaps, a GL-facing use (its tex_id, bind) raises `StaleTextureError`, so a buffer kept
across a swap fails where it is used instead of showing the image of another frame. The owner
itself, used as a texture, is always its current buffer: hold the owner, and take its texture where
it is used.
"""

from typing import Any


class StaleTextureError(RuntimeError):
    """A double buffer's buffer was used after its owner swapped: it holds another frame than it did."""


class SwapHandle:
    """The checking part of a handle. A handle class puts it before the buffer's own class and so
    skips that class's __init__: every field reads through to the buffer, and the buffer class's
    methods, run on the handle, reach the GL through the checked tex_id."""

    def __init__(self, owner: Any, buffer: Any, swaps: int) -> None:
        self._owner: Any = owner
        self._buffer: Any = buffer
        self._swaps: int = swaps

    def _checked(self) -> Any:
        if self._owner._swaps != self._swaps:
            raise StaleTextureError(f"a {type(self._owner).__name__} buffer was used after its owner swapped: hold the "
                                    "owner and read its .texture where it is used, never keep the texture")
        return self._buffer

    @property
    def tex_id(self) -> int:
        return self._checked().tex_id

    def bind(self) -> None:
        self._checked().bind()

    def unbind(self) -> None:
        self._buffer.unbind()

    def __getattr__(self, name: str) -> Any:
        # sizes, formats and flags (fields the buffer's class sets in its __init__), unchecked
        if name in ('_owner', '_buffer', '_swaps'):
            raise AttributeError(name)
        return getattr(self._buffer, name)
