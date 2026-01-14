"""OpenGL state management utilities (OpenFrameworks-inspired)."""
from OpenGL.GL import * # type: ignore
from dataclasses import dataclass


@dataclass
class Style:
    """Current OpenGL rendering style state."""
    blend_enabled: bool = False
    blend_src: int = GL_SRC_ALPHA
    blend_dst: int = GL_ONE_MINUS_SRC_ALPHA
    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    line_width: float = 1.0


_style_stack: list[Style] = []
_MAX_STYLE_HISTORY: int = 32


def pushStyle() -> None:
    """Save current OpenGL rendering state."""
    state = Style(
        blend_enabled=bool(glIsEnabled(GL_BLEND)),
        blend_src=glGetIntegerv(GL_BLEND_SRC),
        blend_dst=glGetIntegerv(GL_BLEND_DST),
        color=tuple(glGetFloatv(GL_CURRENT_COLOR)),
        line_width=glGetFloatv(GL_LINE_WIDTH)[0],
    )
    _style_stack.append(state)

    if len(_style_stack) > _MAX_STYLE_HISTORY:
        _style_stack.pop(0)
        print(f"pushStyle(): maximum style stack depth {_MAX_STYLE_HISTORY} reached")


def popStyle() -> None:
    """Restore previously saved OpenGL rendering state."""
    if not _style_stack:
        print("popStyle() called without matching pushStyle()")
        return

    state = _style_stack.pop()

    if state.blend_enabled:
        glEnable(GL_BLEND)
    else:
        glDisable(GL_BLEND)

    glBlendFunc(state.blend_src, state.blend_dst)
    glColor4f(*state.color)
    glLineWidth(state.line_width)


def setOrthoView(width: int, height: int, flip_y: bool = True) -> None:
    """Set up orthographic projection for 2D rendering.

    Args:
        width: Viewport width in pixels
        height: Viewport height in pixels
        flip_y: If True, origin at top-left (UI convention).
                If False, origin at bottom-left (OpenGL convention)
    """
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    if flip_y:
        glOrtho(0, width, height, 0, -1, 1)
    else:
        glOrtho(0, width, 0, height, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glViewport(0, 0, width, height)