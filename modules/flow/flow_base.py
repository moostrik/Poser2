"""Base class for flow processing layers.

Ported from ofxFlowTools ftFlow.h/cpp
"""

from abc import ABC

from modules.gl import SwapFbo, Texture
from .flow_util import FlowUtil


class FlowBase(ABC):
    """Base class for flow processing with input/output FBOs.

    Provides input_fbo and output_fbo (SwapFbo) for ping-pong rendering.

    Derived classes must:
    1. Set _input_internal_format and _output_internal_format in __init__()
    2. Implement update(delta_time) to process input_fbo → output_fbo
    3. Expose domain-specific public APIs (e.g., set_velocity(), .velocity property)
    4. Never expose _input, _output, _set(), or _add() directly

    Protected methods (_set, _add) and properties (_input, _output) are for
    internal use only. Derived classes wrap these with semantic names.
    """

    def __init__(self) -> None:
        self._input_fbo: SwapFbo = SwapFbo()
        self._output_fbo: SwapFbo = SwapFbo()
        self._allocated: bool = False

        # Subclasses must set these in __init__
        self._input_internal_format: int = 0
        self._output_internal_format: int = 0

    @property
    def _input(self) -> Texture:
        """Protected: the input buffer, the `SwapFbo` itself, so it is always its current buffer and
        can be held across a swap. Access via domain-specific properties in derived classes."""
        return self._input_fbo

    @property
    def _output(self) -> Texture:
        """Protected: the output buffer, the `SwapFbo` itself, so it is always its current buffer and
        can be held across a swap. Access via domain-specific properties in derived classes."""
        return self._output_fbo

    @property
    def allocated(self) -> bool:
        """Check if buffers are allocated."""
        return self._allocated

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        """Allocate input/output FBOs using formats set by derived class.

        Args:
            width: Input buffer width
            height: Input buffer height
            output_width: Output buffer width (defaults to width)
            output_height: Output buffer height (defaults to height)
        """
        if self._input_internal_format == 0 or self._output_internal_format == 0:
            raise RuntimeError(f"{self.__class__.__name__} must set input_internal_format and output_internal_format in __init__")

        out_w = output_width if output_width is not None else width
        out_h = output_height if output_height is not None else height

        self._input_fbo.allocate(width, height, self._input_internal_format)
        FlowUtil.zero(self._input_fbo)

        self._output_fbo.allocate(out_w, out_h, self._output_internal_format)
        FlowUtil.zero(self._output_fbo)

        self._allocated = True

    def deallocate(self) -> None:
        """Release all FBO resources."""
        self._input_fbo.deallocate()
        self._output_fbo.deallocate()
        self._allocated = False

    # def _set(self, texture: Texture, strength: float = 1.0) -> None:
    #     """Protected: Replace input buffer. Wrap with semantic method in derived class."""
    #     if not self._allocated:
    #         return
    #     FlowUtil.set(self._input_fbo, texture, strength)

    # def _add(self, texture: Texture, strength: float = 1.0) -> None:
    #     """Protected: Add to input buffer. Wrap with semantic method in derived class."""
    #     if not self._allocated:
    #         return
    #     FlowUtil.add(self._input_fbo, texture, strength)

    def reset(self) -> None:
        """Clear input and output buffers to zero."""
        FlowUtil.zero(self._input_fbo)
        FlowUtil.zero(self._output_fbo)

