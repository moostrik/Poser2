"""The blend state of the GL context made current, kept on the application side (as Cinder's blend state).

Each GL context holds its own `BlendStack` (`gl.Context`); this module's functions work on the current one's, so the
state it holds is that context's and no `glGet*` is asked. Every blend change goes through this module.
"""
from OpenGL.GL import GL_BLEND, GL_DST_COLOR, GL_FUNC_ADD, GL_MAX, GL_MIN, GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE_MINUS_SRC_COLOR, GL_SRC_ALPHA, GL_ZERO, glBlendEquationi, glBlendFunc, glBlendFunci, glDisable, glEnable
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class BlendMode(Enum):
    """Common OpenGL blend modes for intuitive blending control."""

    ALPHA = (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # Standard transparency
    ADD = (GL_SRC_ALPHA, GL_ONE)  # Additive blending (brightens)
    MULTIPLY = (GL_DST_COLOR, GL_ZERO)  # Multiply blending (darkens)
    SCREEN = (GL_ONE, GL_ONE_MINUS_SRC_COLOR)  # Screen blending (lightens)
    SUM = (GL_ONE, GL_ONE)  # Plain sum of every channel, alpha included (accumulation targets)
    DISABLED = (GL_ONE, GL_ZERO)  # No blending (replaces destination)
    PREMULTIPLIED = (GL_ONE, GL_ONE_MINUS_SRC_ALPHA)  # Transparency of a color already times its alpha
    REVEALAGE = (GL_ZERO, GL_ONE_MINUS_SRC_COLOR)  # The destination times one minus the source, per channel

    @property
    def premultiplied(self) -> 'BlendMode':
        """The same blend for a color already times its alpha: its alpha not applied again."""
        return {BlendMode.ALPHA: BlendMode.PREMULTIPLIED, BlendMode.ADD: BlendMode.SUM}.get(self, self)

    @property
    def src_factor(self) -> int:
        """Get the source blend factor."""
        return self.value[0]

    @property
    def dst_factor(self) -> int:
        """Get the destination blend factor."""
        return self.value[1]


class BlendEquation(Enum):
    """How a blend combines the source and the destination of a draw buffer."""
    ADD = GL_FUNC_ADD       # the factored terms summed: every blend mode
    MIN = GL_MIN            # the smaller of the two, the factors ignored
    MAX = GL_MAX            # the larger of the two, the factors ignored


@dataclass
class BlendState:
    """A blend state: on or off, the mode of every draw buffer, the buffers set to another mode, and the buffers set to
    another equation than ADD."""
    blend_enabled: bool = False
    blend_mode: BlendMode = BlendMode.ALPHA
    buffer_modes: dict[int, BlendMode] = field(default_factory=dict)
    buffer_equations: dict[int, BlendEquation] = field(default_factory=dict)

    def copy(self) -> 'BlendState':
        return BlendState(self.blend_enabled, self.blend_mode, dict(self.buffer_modes), dict(self.buffer_equations))


_MAX_SAVED: int = 32


class BlendStack:
    """One GL context's blend state and the states saved by `push`."""

    def __init__(self) -> None:
        self.current: BlendState = BlendState()
        self.saved: list[BlendState] = []

    def reset(self) -> None:
        """GL's defaults, blending off and the alpha blend set, in the context made current for this stack."""
        self.current = BlendState()
        self.saved.clear()
        glDisable(GL_BLEND)
        glBlendFunc(BlendMode.ALPHA.src_factor, BlendMode.ALPHA.dst_factor)


_stack: BlendStack = BlendStack()       # until a context is made current


def use(stack: BlendStack) -> None:
    """Work on `stack` from now on: the context it belongs to is made current (`gl.Context.make_current`)."""
    global _stack
    _stack = stack


def push() -> None:
    """Save current rendering state (from shadow state, no GPU query)."""
    _stack.saved.append(_stack.current.copy())
    if len(_stack.saved) > _MAX_SAVED:
        _stack.saved.pop(0)
        logger.warning("push(): at most %s blend states saved; the oldest is dropped", _MAX_SAVED)


def pop() -> None:
    """Restore previously saved rendering state."""
    if not _stack.saved:
        logger.warning("pop() called without matching push()")
        return

    state: BlendState = _stack.saved.pop()
    current: BlendState = _stack.current

    # Only issue GL calls if state actually changed
    if state.blend_enabled != current.blend_enabled:
        if state.blend_enabled:
            glEnable(GL_BLEND)
        else:
            glDisable(GL_BLEND)

    if state.blend_mode != current.blend_mode or state.buffer_modes != current.buffer_modes:
        glBlendFunc(state.blend_mode.src_factor, state.blend_mode.dst_factor)
        for buffer, mode in state.buffer_modes.items():
            glBlendFunci(buffer, mode.src_factor, mode.dst_factor)

    for buffer in set(state.buffer_equations) | set(current.buffer_equations):
        equation: BlendEquation = state.buffer_equations.get(buffer, BlendEquation.ADD)
        if equation != current.buffer_equations.get(buffer, BlendEquation.ADD):
            glBlendEquationi(buffer, equation.value)

    _stack.current = state


def set_blend_mode(mode: BlendMode) -> None:
    """Set the blend mode of every draw buffer (with redundant state change elimination)."""
    current: BlendState = _stack.current
    new_enabled = (mode != BlendMode.DISABLED)

    # Only issue GL calls if state actually changes
    if new_enabled != current.blend_enabled:
        if new_enabled:
            glEnable(GL_BLEND)
        else:
            glDisable(GL_BLEND)
        current.blend_enabled = new_enabled

    if mode != current.blend_mode or current.buffer_modes:
        glBlendFunc(mode.src_factor, mode.dst_factor)
        current.blend_mode = mode
        current.buffer_modes = {}


def set_buffer_blend_equation(buffer: int, equation: BlendEquation) -> None:
    """Set the blend equation of draw buffer `buffer`; `pop` restores the one before."""
    current: BlendState = _stack.current
    if current.buffer_equations.get(buffer, BlendEquation.ADD) != equation:
        glBlendEquationi(buffer, equation.value)
        if equation == BlendEquation.ADD:
            current.buffer_equations.pop(buffer, None)
        else:
            current.buffer_equations[buffer] = equation


def set_buffer_blend_mode(buffer: int, mode: BlendMode) -> None:
    """Set the blend mode of draw buffer `buffer` alone; `set_blend_mode` sets every buffer again."""
    current: BlendState = _stack.current
    if current.buffer_modes.get(buffer, current.blend_mode) != mode:
        glBlendFunci(buffer, mode.src_factor, mode.dst_factor)
        current.buffer_modes[buffer] = mode
