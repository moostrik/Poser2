"""The cut: a volume laid flat into a 2D texture: the slice at one layer, or the projection along z
(front), y (top) or x (side), in the box's proportions, so a layer gap looks as deep as it is. The
monitor's strip reads the front; the harnesses read layers back.

The sizes and the projection rule are plain numbers, so they test without GL."""

from enum import IntEnum, auto

from OpenGL.GL import GL_RGBA16F

from modules.gl import Fbo, Texture, Texture3D
from ..port import TextureKind
from .shaders import VolumeCutShader


class Cut(IntEnum):
    """A volume laid flat: one layer, or the projection seen from the front, the top or the side. The
    monitor's strip reads the front; the harnesses read layers back."""
    SLICE = 0
    FRONT = auto()
    TOP = auto()
    SIDE = auto()


def cut_size(cut: Cut, width: int, height: int, extent: float) -> tuple[int, int]:
    """The texels of a cut of a volume `width` × `height` texels across in a box `extent` widths
    deep: the face for a slice or the front, width × depth for the top, depth × height for the side."""
    depth: int = max(1, round(extent * width))
    if cut == Cut.TOP:
        return (width, depth)
    if cut == Cut.SIDE:
        return (depth, height)
    return (width, height)


def projects_max(kind: TextureKind) -> bool:
    """Whether a projection takes the max: for an image, whose values are amounts; a velocity's and a
    scalar's signs would cancel in a max, so they take the mean."""
    return kind == TextureKind.IMAGE


class VolumeCut:
    """Draws a volume's flat cut into its own FBO (RGBA16F, the values as they are); render thread only."""

    def __init__(self) -> None:
        self._fbo: Fbo = Fbo()
        self._shader: VolumeCutShader = VolumeCutShader()

    @property
    def texture(self) -> Texture:
        """The cut as drawn last."""
        return self._fbo

    def allocate(self) -> None:
        self._shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._shader.deallocate()

    def process(self, volume: Texture3D, cut: Cut, position: float, kind: TextureKind, extent: float) -> Texture:
        """Draw the flat cut of `volume` at `position` (0 front, 1 back, for a slice); returns its texture."""
        size: tuple[int, int] = cut_size(cut, volume.width, volume.height, extent)
        if not self._fbo.allocated or (self._fbo.width, self._fbo.height) != size:
            self._fbo.deallocate()
            self._fbo.allocate(*size, GL_RGBA16F)
        self._fbo.begin()
        self._shader.reload()
        self._shader.use(volume, int(cut) - int(Cut.SLICE), position, projects_max(kind), kind.vector)
        self._fbo.end()
        return self._fbo
