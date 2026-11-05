"""The bridge stage: the layer that turns the measured velocity into what the fluid injects.

`BridgeInput` is what the app hands it each tick, `BridgeOutput` what it hands out. Both hold values
only: a texture field holds the `Fbo`, `SwapFbo` or `Tensor` object itself, never a buffer taken
from it.

`BridgeLayer` is the shape the 2D and the 3D bridge layer share: it steps its bridges every tick on
the input's `dt` (they are filters in time, so a velocity held between camera frames reads right)
and makes the density's color from the image, the tint and the mask. It draws nothing; the monitors
draw its output's `textures`."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from OpenGL.GL import GL_RGBA16F

from modules.gl import Fbo, Texture, Style
from modules.settings import BaseSettings
from ..layer_base import LayerBase
from ...shaders import DensityColor
from .port import Port


@dataclass
class BridgeInput:
    """One camera's velocity for the bridge stage, built by the app each tick. `velocity` is the
    motion output; `dt` the render tick's; `image`, or None, the picture whose RGB colors the
    density; `color` an RGB tint on it, white by default, the color alone without an image; `mask`
    the player's mask on the view, or None, which gates the image."""
    velocity: Texture
    dt: float
    image: Texture | None = None
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mask: Texture | None = None


@dataclass
class BridgeOutput:
    """The bridge stage's textures, what the fluid injects: `velocity`; `density`, in 2D the amount
    alone, in 3D RGBA, the amount times the color, with the amount as alpha; `temperature`; `onset`,
    a second velocity, or None; `spin`, a curl at its `spin` setting, or None; and `textures`, its
    ports for the monitors."""
    velocity: Texture
    density: Texture
    temperature: Texture
    onset: Texture | None = None
    spin: Texture | None = None
    textures: dict[str, Port] = field(default_factory=dict)


class BridgeLayerSettings(BaseSettings):
    """What the 2D and the 3D bridge layer share; each adds its bridges' groups."""


class BridgeLayer(LayerBase, ABC):
    """The bridge stage's layer. `set_input` before the render loop; `update` makes the density's
    color and runs the bridges (`_bridge`); `output` is the stage's textures. The bridges run at the
    velocity's size, which the layer follows, the density at the layer's own size. The layer draws nothing."""

    def __init__(self, settings: BridgeLayerSettings) -> None:
        self.settings: BridgeLayerSettings = settings
        self._input: BridgeInput | None = None
        self._color: Fbo = Fbo()                        # the density's color: the image's RGB, or white, times the tint and the mask
        self._color_shader: DensityColor = DensityColor()
        self._flow_size: tuple[int, int] = (0, 0)       # the size the bridges run at: the velocity's
        self._output_size: tuple[int, int] = (0, 0)     # the layer's allocation size

    def set_input(self, input: BridgeInput) -> None:
        """What the layer bridges at its next update."""
        self._input = input

    @property
    @abstractmethod
    def output(self) -> BridgeOutput:
        """The stage's textures, allocated once; what the fluid's injections and the monitors' inputs are built from."""
        ...

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        self._output_size = (width, height)
        self._color.allocate(width, height, GL_RGBA16F)
        self._color_shader.allocate()

    def deallocate(self) -> None:
        if self._flow_size != (0, 0):
            self._deallocate_stages()
            self._flow_size = (0, 0)
        self._color.deallocate()
        self._color_shader.deallocate()

    def reset(self) -> None:
        """Start the bridges from rest."""
        self._reset_stages()

    def _follow_flow_size(self, velocity: Texture) -> None:
        """Allocate the bridges at the velocity's size, and again when it changes."""
        size: tuple[int, int] = (velocity.width, velocity.height)
        if size == self._flow_size:
            return
        if self._flow_size != (0, 0):
            self._deallocate_stages()
        self._flow_size = size
        self._allocate_stages(*size, *self._output_size)
        self._reset_stages()

    @abstractmethod
    def _allocate_stages(self, width: int, height: int, output_width: int, output_height: int) -> None:
        """Allocate the bridges at the velocity's size; `output_width` by `output_height` is the layer's."""

    @abstractmethod
    def _deallocate_stages(self) -> None: ...

    @abstractmethod
    def _reset_stages(self) -> None: ...

    @abstractmethod
    def _bridge(self, input: BridgeInput, color: Texture) -> None:
        """Run the bridges on `input.velocity`, stepping `input.dt`; `color` is the density's color."""

    # ========== Processing ==========

    def update(self) -> None:
        input: BridgeInput | None = self._input
        if input is None or not input.velocity.allocated:
            return
        self._follow_flow_size(input.velocity)

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)

        self._color.begin()
        self._color_shader.use(input.image, input.color, input.mask)
        self._color.end()
        self._bridge(input, self._color)

        Style.pop_style()

    def draw(self) -> None:
        """A stage draws nothing; a monitor draws its textures."""
