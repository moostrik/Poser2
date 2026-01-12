"""Base class for flow processing layers.

Ported from ofxFlowTools ftFlow.h/cpp
"""

from abc import ABC, abstractmethod
from OpenGL.GL import *  # type: ignore

from modules.gl import SwapFbo, Texture
from .FlowUtil import FlowUtil
from modules.utils.PointsAndRects import Rect


class FlowBase(ABC):
    """Base class for flow processing with input/output FBOs.

    All flow layers have:
    - input_fbo: SwapFbo for receiving data
    - output_fbo: SwapFbo for producing results

    Subclasses define internal formats in __init__() by setting:
        self.input_internal_format = GL_R8
        self.output_internal_format = GL_RG32F

    Then call allocate() to create FBOs with those formats.
    """

    def __init__(self) -> None:
        self.input_fbo: SwapFbo = SwapFbo()
        self.output_fbo: SwapFbo = SwapFbo()
        self._allocated: bool = False

        # Subclasses must set these in __init__
        self.input_internal_format: int = 0
        self.output_internal_format: int = 0

    @property
    def input(self) -> Texture:
        """Input texture."""
        return self.input_fbo.texture

    @property
    def output(self) -> Texture:
        """Output texture."""
        return self.output_fbo.texture

    @property
    def allocated(self) -> bool:
        """Check if layer is allocated."""
        return self._allocated

    def allocate(self, width: int, height: int, output_width: int | None = None, output_height: int | None = None) -> None:
        """Allocate input/output FBOs using internal formats from subclass.

        Args:
            width: Input width (or width for both if outputs not specified)
            height: Input height (or height for both if outputs not specified)
            output_width: Optional output width (defaults to width)
            output_height: Optional output height (defaults to height)
        """
        if self.input_internal_format == 0 or self.output_internal_format == 0:
            raise RuntimeError(f"{self.__class__.__name__} must set input_internal_format and output_internal_format in __init__")

        out_w = output_width if output_width is not None else width
        out_h = output_height if output_height is not None else height

        self.input_fbo.allocate(width, height, self.input_internal_format)
        FlowUtil.zero(self.input_fbo)

        self.output_fbo.allocate(out_w, out_h, self.output_internal_format)
        FlowUtil.zero(self.output_fbo)

        self._allocated = True

    def deallocate(self) -> None:
        """Release all FBO resources."""
        self.input_fbo.deallocate()
        self.output_fbo.deallocate()
        self._allocated = False

    def set(self, texture: Texture) -> None:
        """Set input texture (copies to input FBO).

        Args:
            texture: Input texture
        """
        if not self._allocated:
            return
        FlowUtil.stretch(self.input_fbo, texture)

    def add(self, texture: Texture, strength: float = 1.0) -> None:
        """Add to input FBO with strength multiplier.

        Args:
            texture: Input texture
            strength: Blend strength
        """
        if not self._allocated:
            return
        FlowUtil.add(self.input_fbo, texture, strength)

    def reset(self) -> None:
        """Reset both input and output FBOs to zero."""
        FlowUtil.zero(self.input_fbo)
        FlowUtil.zero(self.output_fbo)

    @abstractmethod
    def update(self) -> None:
        """Process input and generate output. Must be implemented by subclass."""
        ...

    def draw(self, rect: Rect) -> None:
        """Draw output FBO to screen."""
        self.draw_output(rect)

    def draw_input(self, rect: Rect) -> None:
        """Draw input FBO to screen."""
        self.input_fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def draw_output(self, rect: Rect) -> None:
        """Draw output FBO to screen."""
        self.output_fbo.draw(rect.x, rect.y, rect.width, rect.height)
