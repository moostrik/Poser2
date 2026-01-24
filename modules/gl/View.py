from OpenGL.GL import * # type: ignore

_viewport_stack: list[tuple[int, int, int, int]] = []
_current_viewport: tuple[int, int, int, int] = (0, 0, 1, 1)

def set_view(w: int, h: int) -> None:
    # return
    global _current_viewport
    _current_viewport = (0, 0, w, h)
    glViewport(0, 0, w, h)

def push_view() -> None:
    # return
    _viewport_stack.append(_current_viewport)

def pop_view() -> None:
    global _current_viewport
    if _viewport_stack:
        _current_viewport = _viewport_stack.pop()
        glViewport(*_current_viewport)
