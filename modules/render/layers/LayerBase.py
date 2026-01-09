# Standard library imports
from abc import ABC, abstractmethod

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.utils.PointsAndRects import Rect
from modules.gl import Texture


class LayerBase(ABC):
    """Base class for all rendering layers.

    Defines the core interface that all layers must implement.
    Use TextureLayer for layers that produce texture output,
    or DirectDrawLayer for layers that render primitives directly.
    """

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        pass

    @abstractmethod
    def deallocate(self) -> None: ...

    @abstractmethod
    def update(self) -> None: ...

    @abstractmethod
    def draw(self, rect: Rect) -> None: ...


class TextureLayer(LayerBase):
    """Base class for layers that produce texture output.

    Use for layers that render to FBOs or manage image textures.
    Examples: CentreCamLayer, CamImageRenderer, CentreMaskLayer
    """

    @property
    @abstractmethod
    def texture(self) -> Texture:
        """Output texture of this layer."""
        ...

    def draw(self, rect: Rect) -> None:
        """Default implementation: draw texture to rect."""
        self.texture.draw(rect.x, rect.y, rect.width, rect.height)


class DirectDrawLayer(LayerBase):
    """Base class for layers that render directly without texture output.

    Use for layers that draw GL primitives or perform computation only.
    Examples: CamBBoxRenderer, CentreGeometry
    """
    pass

