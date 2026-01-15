"""Base class for field visualization.

Ported from ofxFlowTools ftVisualization.h
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from OpenGL.GL import *  # type: ignore

from modules.gl import Fbo, Texture
from modules.utils.PointsAndRects import Rect

from .. import FlowConfigBase


@dataclass
class VisualisationFieldConfig(FlowConfigBase):
    """Base configuration for field visualizations."""
    toggle_scalar: bool = field(
        default=False,
        metadata={"label": "Toggle Scalar",
                  "description": "Toggle between direction map and arrow field"}
    )
    scale: float = field(
        default=1.0,
        metadata={"min": 0.0, "max": 10.0, "label": "Scale",
                  "description": "Visualization scale multiplier"}
    )
    spacing: float = field(
        default=8.0,
        metadata={"min": 4.0, "max": 16.0, "label": "Grid Spacing", "description": "Distance between elements (pixels)"}
    )
    element_length: float = field(
        default=8.0,
        metadata={"min": 0.5, "max": 64.0, "label": "Element Length", "description": "Element length in pixels"}
    )
    element_width: float = field(
        default=0.8,
        metadata={"min": 0.5, "max": 2.5, "label": "Element Width", "description": "Element line width"}
    )


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
    def width(self) -> int:
        """Visualization width."""
        return self._width

    @property
    def height(self) -> int:
        """Visualization height."""
        return self._height

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

    def draw(self, rect: Rect) -> None:
        """Draw visualization to screen.

        Args:
            rect: Draw rectangle
        """
        if self._allocated:
            self._fbo.draw(rect.x, rect.y, rect.width, rect.height)
