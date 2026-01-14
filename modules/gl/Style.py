"""OpenGL state management utilities (OpenFrameworks-inspired)."""
from OpenGL.GL import * # type: ignore
from dataclasses import dataclass
from enum import Enum


class BlendMode(Enum):
    """Common OpenGL blend modes for intuitive blending control."""

    ALPHA = (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # Standard transparency
    ADDITIVE = (GL_SRC_ALPHA, GL_ONE)  # Additive blending (brightens)
    MULTIPLY = (GL_DST_COLOR, GL_ZERO)  # Multiply blending (darkens)
    SCREEN = (GL_ONE, GL_ONE_MINUS_SRC_COLOR)  # Screen blending (lightens)
    DISABLED = (GL_ONE, GL_ZERO)  # No blending (replaces destination)

    @property
    def src_factor(self) -> int:
        """Get the source blend factor."""
        return self.value[0]

    @property
    def dst_factor(self) -> int:
        """Get the destination blend factor."""
        return self.value[1]


@dataclass
class _StyleState:
    """Current OpenGL rendering style state."""
    blend_enabled: bool = False
    blend_mode: BlendMode = BlendMode.ALPHA  # Intuitive blend mode
    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    line_width: float = 1.0
    point_size: float = 1.0


_style_stack: list[_StyleState] = []
_MAX_STYLE_HISTORY: int = 32


def push_style() -> None:
    """Save current OpenGL rendering state."""
    # Retrieve current blend factors from OpenGL
    src = glGetIntegerv(GL_BLEND_SRC)
    dst = glGetIntegerv(GL_BLEND_DST)

    # Find matching BlendMode or default to ALPHA
    blend_mode = BlendMode.ALPHA
    for mode in BlendMode:
        if mode.src_factor == src and mode.dst_factor == dst:
            blend_mode = mode
            break

    state = _StyleState(
        blend_enabled=bool(glIsEnabled(GL_BLEND)),
        blend_mode=blend_mode,
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

    glBlendFunc(state.blend_mode.src_factor, state.blend_mode.dst_factor)
    glColor4f(*state.color)
    glLineWidth(state.line_width)
    glPointSize(state.point_size)


def set_blend_mode(mode: BlendMode) -> None:
    """Set the blend mode."""
    if mode == BlendMode.DISABLED:
        glDisable(GL_BLEND)
        glBlendFunc(mode.src_factor, mode.dst_factor)
    else:
        glEnable(GL_BLEND)
        glBlendFunc(mode.src_factor, mode.dst_factor)


def set_color(r: float, g: float, b: float, a: float = 1.0) -> None:
    """Set the current drawing color."""
    glColor4f(r, g, b, a)


def set_line_width(width: float) -> None:
    """Set the line width for line drawing."""
    glLineWidth(width)


def set_point_size(size: float) -> None:
    """Set the point size for point drawing."""
    glPointSize(size)
