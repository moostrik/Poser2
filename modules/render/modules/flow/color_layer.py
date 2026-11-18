"""The color stage: the density's color from the view's image and mask, between the view and the motion.

`ColorInput` is what the app hands it each capture, `ColorOutput` what the bridge takes. The stage runs its one
part, `DensityColor`, at the view's size, made again on a new capture, told by its capture time, or when a setting
it is made with changes (so also while the app holds a frame). It draws nothing; its monitors draw its ports."""

from dataclasses import dataclass
from typing import Annotated

from OpenGL.GL import GL_RGBA16F

from modules.flow import DensityColor, DensityColorSettings, HasPorts, TextureKind
from modules.gl import Texture
from modules.settings import Group
from .monitor_2d_layer import DEBUG_TINT, MAIN_TINT, Monitor2DLayerSettings
from .stage_layer import StageLayer, StageLayerSettings, StageOutput


@dataclass
class ColorInput:
    """One camera's view for the color stage: `image` the view's cropped color, `mask` the player's mask on it,
    `timestamp` the capture's time, its identity (None: no capture yet); `made`, what the image and the mask were
    made from (the view's capture and settings), so the color is made again when they change without a capture."""
    image: Annotated[Texture, TextureKind.IMAGE]
    mask: Annotated[Texture, TextureKind.SCALAR]
    timestamp: float | None
    made: tuple = ()


@dataclass
class ColorOutput(StageOutput):
    """The color stage's texture: `color`, the density's color, RGBA, premultiplied by its alpha, the coverage."""
    color: Annotated[Texture, TextureKind.IMAGE]


class ColorLayerSettings(DensityColorSettings, StageLayerSettings):
    """Configuration for ColorLayer: the reset, the density's color, its monitors last."""
    monitor_main:  Group[Monitor2DLayerSettings] = Group(Monitor2DLayerSettings, last=True, tint=MAIN_TINT)
    monitor_debug: Group[Monitor2DLayerSettings] = Group(Monitor2DLayerSettings, last=True, tint=DEBUG_TINT)


class ColorLayer(StageLayer[ColorInput, ColorOutput]):
    """The color stage's layer, allocated at the view's size. Draws nothing."""

    def __init__(self, settings: ColorLayerSettings) -> None:
        super().__init__(settings)
        self.settings: ColorLayerSettings = settings
        self._color: DensityColor = DensityColor(settings)
        self._made: tuple | None = None      # the capture and the settings the color was made from

    @property
    def output(self) -> ColorOutput:
        return ColorOutput(self._color.texture)

    def _components(self) -> dict[str, HasPorts]:
        return {'color': self._color}

    @property
    def texture(self) -> Texture:
        """The color, the shape the stage's monitor is drawn in."""
        return self._color.texture

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        self._color.allocate(width, height, GL_RGBA16F)
        self._made = None

    def deallocate(self) -> None:
        self._color.deallocate()
        super().deallocate()

    def reset(self) -> None:
        self._color.clear()
        self._made = None

    def draw(self) -> None:
        """A stage draws nothing; a monitor draws its textures."""

    def _update(self) -> None:
        input: ColorInput | None = self._input
        if input is None or input.timestamp is None or not input.image.allocated or not input.mask.allocated:
            return
        made: tuple = (input.timestamp, input.made, input.image.width, input.image.height, self._color.made_with)
        if made == self._made:
            return
        self._made = made
        self._color.process(input.image, input.mask)
