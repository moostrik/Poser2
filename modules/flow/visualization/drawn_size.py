"""The size a view renders at: the size it is drawn at. Plain numbers, so it tests without GL.

A view learns its size only when it is drawn (the viewport the app fits it into), and it renders on
its first draw of a tick, before a second draw at another size is seen. So it renders at the largest
of last tick's viewports, and grows at once when a larger one turns up; a smaller layout takes over
one tick later."""

Size = tuple[int, int]


def _larger(a: Size, b: Size) -> Size:
    return (max(a[0], b[0]), max(a[1], b[1]))


class DrawnSize:
    """`tick` at each update, `see` at each draw; `see` returns the size to render at."""

    def __init__(self) -> None:
        self._target: Size = (0, 0)     # the largest viewport of the tick before
        self._seen: Size = (0, 0)       # the largest viewport of this tick so far
        self._size: Size = (0, 0)       # the size handed out last

    @property
    def size(self) -> Size:
        """The size handed out last; (0, 0) before the first draw."""
        return self._size

    def tick(self) -> None:
        """A new tick: last tick's largest viewport becomes the size to render at."""
        if self._seen != (0, 0):
            self._target = self._seen
        self._seen = (0, 0)

    def see(self, width: int, height: int) -> Size:
        """A draw into a viewport of `width` × `height`; the size to render at, grown to cover it."""
        viewport: Size = (max(int(width), 1), max(int(height), 1))
        first: bool = self._seen == (0, 0)
        self._seen = _larger(self._seen, viewport)
        base: Size = self._target if first else self._size
        self._size = _larger(base, viewport)
        return self._size
