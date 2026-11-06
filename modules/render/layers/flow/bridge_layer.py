"""The bridge stage: the layer that turns the measured velocity into what the fluid injects.

`BridgeInput` is what the app hands it each tick, `BridgeOutput` what it hands out. Both hold values
only: a texture field holds the `Fbo`, `SwapFbo` or `Tensor` object itself, never a buffer taken
from it.

`BridgeLayer` is the shape the 2D and the 3D bridge layer share: it steps its bridges every tick on
the input's `dt` (they are filters in time, so a velocity held between camera frames reads right)
and makes the density's color from the image, the tint and the mask. It draws nothing; the monitors
draw its output's `textures`."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Annotated

from OpenGL.GL import GL_RGBA16F

from modules.gl import Fbo, Texture, Style
from modules.flow import HasPorts, Port, Ports, TextureKind
from modules.settings import Field
from ...shaders import DensityColor
from .stage_layer import StageLayer, StageLayerSettings, StageOutput


@dataclass
class BridgeInput:
    """One camera's velocity for the bridge stage, built by the app each tick. `velocity` is the
    motion output; `dt` the render tick's; `image`, or None, the picture whose RGB colors the
    density; `color` an RGB tint on it, white by default, the color alone without an image; `mask`
    the player's mask on the view, or None, which gates the image."""
    velocity: Annotated[Texture, TextureKind.VELOCITY]
    dt: float
    image: Annotated[Texture | None, TextureKind.IMAGE] = None
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mask: Annotated[Texture | None, TextureKind.SCALAR] = None


@dataclass
class BridgeOutput(StageOutput):
    """The bridge stage's textures, what the fluid injects: `velocity`; `density`, in 2D the amount
    alone, in 3D RGBA, the amount times the color, with the amount as alpha; `temperature`; `onset`,
    a second velocity, or None."""
    velocity: Annotated[Texture, TextureKind.VELOCITY]
    density: Annotated[Texture, TextureKind.IMAGE]
    temperature: Annotated[Texture, TextureKind.SCALAR]
    onset: Annotated[Texture | None, TextureKind.VELOCITY] = None


class BridgeLayerSettings(StageLayerSettings):
    """What the 2D and the 3D bridge layer share; each adds its bridges' groups."""
    width: Field[int] =     Field(2160, min=32, max=4096, step=32, description="Width the bridge writes its outputs at")
    height: Field[int] =    Field(3840, min=32, max=4096, step=32, description="Height the bridge writes its outputs at")


class BridgeLayer(StageLayer[BridgeInput, BridgeOutput]):
    """The bridge stage's layer. `update` makes the density's color and runs the bridges (`_bridge`).
    The bridges' filters run at the velocity's size, which the layer follows; the outputs are written
    at `settings.width` by `settings.height`, which it follows too. The layer draws nothing."""

    def __init__(self, settings: BridgeLayerSettings) -> None:
        super().__init__(settings)
        self.settings: BridgeLayerSettings = settings
        self._color: Fbo = Fbo()                        # the density's color: the image's RGB, or white, times the tint and the mask
        self._color_shader: DensityColor = DensityColor()
        self._flow_size: tuple[int, int] = (0, 0)       # the size the filters run at: the velocity's
        self._output_size: tuple[int, int] = (0, 0)     # the size the outputs are written at: settings.width/height

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        """The size the app gives is not the bridge's: it writes at `settings.width` by `settings.height`."""
        self._color_shader.allocate()

    def deallocate(self) -> None:
        self._deallocate_sizes()
        self._color_shader.deallocate()
        super().deallocate()

    def reset(self) -> None:
        """Start the bridges from rest."""
        self._reset_stages()

    def _follow_sizes(self, velocity: Texture) -> None:
        """Allocate the bridges at the velocity's size and the output size, and again when either changes."""
        flow_size: tuple[int, int] = (velocity.width, velocity.height)
        output_size: tuple[int, int] = (self.settings.width, self.settings.height)
        if (flow_size, output_size) == (self._flow_size, self._output_size):
            return
        self._deallocate_sizes()
        self._flow_size, self._output_size = flow_size, output_size
        self._color.allocate(*output_size, GL_RGBA16F)
        self._allocate_stages(*flow_size, *output_size)
        self._reset_stages()

    def _deallocate_sizes(self) -> None:
        if self._flow_size == (0, 0):
            return
        self._deallocate_stages()
        self._color.deallocate()
        self._flow_size = self._output_size = (0, 0)

    @abstractmethod
    def _allocate_stages(self, width: int, height: int, output_width: int, output_height: int) -> None:
        """Allocate the bridges' filters at the velocity's size and their outputs at `output_width` by `output_height`."""

    @abstractmethod
    def _deallocate_stages(self) -> None: ...

    @abstractmethod
    def _reset_stages(self) -> None: ...

    @abstractmethod
    def _bridge(self, input: BridgeInput, color: Texture) -> None:
        """Run the bridges on `input.velocity`, stepping `input.dt`; `color` is the density's color."""

    @abstractmethod
    def _bridges(self) -> dict[str, HasPorts]:
        """The bridges it runs, by their settings group's name, in the order it runs them."""

    def _components(self) -> dict[str, HasPorts]:
        """The density's color, made first, then the bridges."""
        return {'': Ports({'color': Port(self._color, TextureKind.IMAGE)}), **self._bridges()}

    # ========== Processing ==========

    def _update(self) -> None:
        input: BridgeInput | None = self._input
        if input is None or not input.velocity.allocated:
            return
        self._follow_sizes(input.velocity)

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        self._color.begin()
        self._color_shader.use(input.image, input.color, input.mask)
        self._color.end()
        self._bridge(input, self._color)

        Style.pop_style()

    def draw(self) -> None:
        """A stage draws nothing; a monitor draws its textures."""
