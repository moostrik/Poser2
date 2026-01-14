"""OpenGL state management utilities (OpenFrameworks-inspired)."""
from OpenGL.GL import * # type: ignore
from dataclasses import dataclass


@dataclass
class _StyleState:
    """Current OpenGL rendering style state."""
    blend_enabled: bool = False
    blend_src: int = GL_SRC_ALPHA
    blend_dst: int = GL_ONE_MINUS_SRC_ALPHA
    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    line_width: float = 1.0
    point_size: float = 1.0


_style_stack: list[_StyleState] = []
_MAX_STYLE_HISTORY: int = 32


def push_style() -> None:
    """Save current OpenGL rendering state."""
    state = _StyleState(
        blend_enabled=bool(glIsEnabled(GL_BLEND)),
        blend_src=glGetIntegerv(GL_BLEND_SRC),
        blend_dst=glGetIntegerv(GL_BLEND_DST),
        color=tuple(glGetFloatv(GL_CURRENT_COLOR)),
        line_width=glGetFloatv(GL_LINE_WIDTH)[0],
        point_size=glGetFloatv(GL_POINT_SIZE)[0],
    )
    _style_stack.append(state)

    if len(_style_stack) > _MAX_STYLE_HISTORY:
        _style_stack.pop(0)
        print(f"push_style(): maximum style stack depth {_MAX_STYLE_HISTORY} reached")


def pop_style() -> None:
    """Restore previously saved OpenGL rendering state."""
    if not _style_stack:
        print("pop_style() called without matching push_style()")
        return

    state = _style_stack.pop()

    if state.blend_enabled:
        glEnable(GL_BLEND)
    else:
        glDisable(GL_BLEND)

    glBlendFunc(state.blend_src, state.blend_dst)
    glColor4f(*state.color)
    glLineWidth(state.line_width)
    glPointSize(state.point_size)
