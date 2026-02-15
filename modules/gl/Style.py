"""OpenGL state management utilities (OpenFrameworks-inspired).

Application-side state tracking to avoid expensive glGet* calls that cause CPU-GPU sync stalls.
All state changes go through this module to keep shadow state in sync with actual GL state.
"""
from OpenGL.GL import * # type: ignore
from dataclasses import dataclass
from enum import Enum


class BlendMode(Enum):
    """Common OpenGL blend modes for intuitive blending control."""

    ALPHA = (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # Standard transparency
    ADD = (GL_SRC_ALPHA, GL_ONE)  # Additive blending (brightens)
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
    blend_mode: BlendMode = BlendMode.ALPHA


# Application-side shadow state (avoids glGet* queries)
_current_state = _StyleState(blend_enabled=False, blend_mode=BlendMode.ALPHA)
_style_stack: list[_StyleState] = []
_MAX_STYLE_HISTORY: int = 32


def reset_state() -> None:
    """Reset shadow state to match OpenGL defaults. Call at start of frame or after context switch."""
    global _current_state
    _current_state = _StyleState(blend_enabled=False, blend_mode=BlendMode.ALPHA)
    _style_stack.clear()
    # Sync GL state to match
    glDisable(GL_BLEND)
    glBlendFunc(BlendMode.ALPHA.src_factor, BlendMode.ALPHA.dst_factor)


def push_style() -> None:
    """Save current rendering state (from shadow state, no GPU query)."""
    global _current_state

    # Copy current shadow state to stack
    state = _StyleState(
        blend_enabled=_current_state.blend_enabled,
        blend_mode=_current_state.blend_mode,
    )
    _style_stack.append(state)

    if len(_style_stack) > _MAX_STYLE_HISTORY:
        _style_stack.pop(0)
        print(f"push_style(): maximum style stack depth {_MAX_STYLE_HISTORY} reached")


def pop_style() -> None:
    """Restore previously saved rendering state."""
    global _current_state

    if not _style_stack:
        print("pop_style() called without matching push_style()")
        return

    state = _style_stack.pop()

    # Only issue GL calls if state actually changed
    if state.blend_enabled != _current_state.blend_enabled:
        if state.blend_enabled:
            glEnable(GL_BLEND)
        else:
            glDisable(GL_BLEND)

    if state.blend_mode != _current_state.blend_mode:
        glBlendFunc(state.blend_mode.src_factor, state.blend_mode.dst_factor)

    _current_state = state


def set_blend_mode(mode: BlendMode) -> None:
    """Set the blend mode (with redundant state change elimination)."""
    global _current_state

    new_enabled = (mode != BlendMode.DISABLED)

    # Only issue GL calls if state actually changes
    if new_enabled != _current_state.blend_enabled:
        if new_enabled:
            glEnable(GL_BLEND)
        else:
            glDisable(GL_BLEND)
        _current_state.blend_enabled = new_enabled

    if mode != _current_state.blend_mode:
        glBlendFunc(mode.src_factor, mode.dst_factor)
        _current_state.blend_mode = mode

