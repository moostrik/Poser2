"""Base class for field visualization.

Ported from ofxFlowTools ftVisualization.h
"""
from abc import ABC, abstractmethod

from OpenGL.GL import *  # type: ignore

from modules.gl import Fbo, Texture, Blit
from modules.utils.PointsAndRects import Rect

from modules.settings import Field, Settings


class VisualisationFieldSettings(Settings):
    """Base configuration for field visualizations."""
    toggle_scalar = Field(False, description="Toggle between direction map and arrow field")
    scale = Field(1.0, min=0.0, max=10.0, description="Visualization scale multiplier")
    spacing = Field(20.0, min=4.0, max=64.0, description="Distance between elements (pixels)")
    element_length = Field(40.0, description="Arrow element length")
    element_width = Field(1.0, min=0.5, max=2.5, description="Element line width")


class FieldBase(ABC):
    """Base class for field visualization renderers.

    Simple rendering wrapper (texture → render → draw), not a processing pipeline.
    Subclasses implement specific visualization methods (velocity, temperature, etc.).

    Ported from ofxFlowTools ftVisualization.h
    """

    def __init__(self) -> None:
        self._fbo: Fbo = Fbo()
        self._allocated: bool = False
        self._width: int = 0
        self._height: int = 0

    @property
    def allocated(self) -> bool:
        """Check if visualization is allocated."""
        return self._allocated

    @property
    def texture(self) -> Texture:
        """Output texture."""
        return self._fbo

    def allocate(self, width: int, height: int) -> None:
        """Allocate visualization resources.

        Args:
            width: Visualization width
            height: Visualization height
        """
        self._width = width
        self._height = height
        self._fbo.allocate(width, height, GL_RGBA8)
        self._allocated = True

    def deallocate(self) -> None:
        """Release all resources."""
        self._fbo.deallocate()
        self._allocated = False

    @abstractmethod
    def set(self, texture: Texture) -> None:
        """Set input texture to visualize.

        Args:
            texture: Input texture
        """
        ...

    @abstractmethod
    def update(self) -> None:
        """Render visualization to internal FBO."""
        ...

    def draw(self) -> None:
        """Draw visualization to screen."""
        if self._allocated:
            Blit.use(self._fbo)
