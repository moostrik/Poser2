"""The bridge stage: the layer that turns the measured velocity into what the fluid injects.

`BridgeInput` is what the app hands it each tick, `BridgeOutput` what it hands out. Both hold values
only: a texture field holds the `Fbo`, `SwapFbo` or `Tensor` object itself, never a buffer taken
from it.

`BridgeLayer` is the shape the 2D and the 3D bridge layer share: it steps its bridges on each update
by the input's `dt` (they are filters in time), at the velocity's size, which it follows. The app
updates it once per camera capture, `dt` the interval between captures, or once per render tick, `dt`
the tick's. It draws nothing; the monitors draw its output's `textures`."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Annotated

from modules.gl import Texture, BlendState
from modules.flow import TextureKind
from modules.settings import Field, Group
from .monitor_2d_layer import DEBUG_TINT, MAIN_TINT, Monitor2DLayerSettings
from .stage_layer import StageLayer, StageLayerSettings, StageOutput


@dataclass
class BridgeInput:
    """One camera's velocity for the bridge stage, built by the app for each step. `velocity` is the
    motion output; `dt` the time the step covers (s); `image`, or None, the picture that colors the
    density, the `color` stage's, taken as it comes; `mask` the player's mask on the view, or None, a share of the
    amounts; `z_position` the motion's, the 3D's only, or None.
    `same_tick`: this step is another capture in the render tick of the step before, so the fluid, which
    steps once a tick, takes per pixel the density, its color and the temperature of the step whose amount is
    larger, and the velocity of this one."""
    velocity: Annotated[Texture, TextureKind.VELOCITY]
    dt: float
    image: Annotated[Texture | None, TextureKind.IMAGE] = None
    mask: Annotated[Texture | None, TextureKind.SCALAR] = None
    z_position: Annotated[Texture | None, TextureKind.Z_POSITION] = None
    same_tick: bool = False


@dataclass
class BridgeOutput(StageOutput):
    """The bridge stage's textures, what the fluid injects, in two systems. The boundary, what the water is
    dragged toward: `velocity`, widths/s; `density`, the amount; `color`, the density's color, RGBA,
    premultiplied by its alpha, the coverage; `temperature`. The body force, what is
    added × dt: `acceleration`, widths/s², None while the bridge gives none.
    The `expansion`, where the water swells, a presence 0 to 1, or None. Where each goes in depth,
    the 3D bridge's only: each output's `_z_position`, its readings' z positions; None without."""
    velocity: Annotated[Texture, TextureKind.VELOCITY]
    density: Annotated[Texture, TextureKind.SCALAR]
    color: Annotated[Texture, TextureKind.IMAGE]
    temperature: Annotated[Texture, TextureKind.SCALAR]
    acceleration: Annotated[Texture | None, TextureKind.ACCELERATION] = None
    velocity_z_position: Annotated[Texture | None, TextureKind.Z_POSITION] = None
    density_z_position: Annotated[Texture | None, TextureKind.Z_POSITION] = None
    temperature_z_position: Annotated[Texture | None, TextureKind.Z_POSITION] = None
    acceleration_z_position: Annotated[Texture | None, TextureKind.Z_POSITION] = None
    expansion: Annotated[Texture | None, TextureKind.SCALAR] = None
    expansion_z_position: Annotated[Texture | None, TextureKind.Z_POSITION] = None


class BridgeLayerSettings(StageLayerSettings):
    """What the 2D and the 3D bridge layer share; each adds its bridges' groups."""
    width: Field[int] =     Field(2160, min=32, max=4096, step=32, description="Width the bridge writes its outputs at")
    height: Field[int] =    Field(3840, min=32, max=4096, step=32, description="Height the bridge writes its outputs at")

    monitor_main:  Group[Monitor2DLayerSettings] = Group(Monitor2DLayerSettings, last=True, tint=MAIN_TINT)
    monitor_debug: Group[Monitor2DLayerSettings] = Group(Monitor2DLayerSettings, last=True, tint=DEBUG_TINT)


class BridgeLayer(StageLayer[BridgeInput, BridgeOutput]):
    """The bridge stage's layer. `update` runs the bridges (`_bridge`) on the input. The bridges' filters run
    at the velocity's size, which the layer follows; the outputs are written at `settings.width` by
    `settings.height`, which it follows too. The layer draws nothing."""

    def __init__(self, settings: BridgeLayerSettings) -> None:
        super().__init__(settings)
        self.settings: BridgeLayerSettings = settings
        self._flow_size: tuple[int, int] = (0, 0)       # the size the filters run at: the velocity's
        self._output_size: tuple[int, int] = (0, 0)     # the size the outputs are written at: settings.width/height

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        """The size the app gives is not the bridge's: it follows the velocity's and writes at `settings.width`
        by `settings.height`, allocating on its first update."""

    def deallocate(self) -> None:
        self._deallocate_sizes()
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
        self._allocate_stages(*flow_size, *output_size)
        self._reset_stages()

    def _deallocate_sizes(self) -> None:
        if self._flow_size == (0, 0):
            return
        self._deallocate_stages()
        self._flow_size = self._output_size = (0, 0)

    @abstractmethod
    def _allocate_stages(self, width: int, height: int, output_width: int, output_height: int) -> None:
        """Allocate the bridges' filters at the velocity's size and their outputs at `output_width` by `output_height`."""

    @abstractmethod
    def _deallocate_stages(self) -> None: ...

    @abstractmethod
    def _reset_stages(self) -> None: ...

    @abstractmethod
    def _bridge(self, input: BridgeInput) -> None:
        """Run the bridges on `input.velocity`, stepping `input.dt`, the density's color first."""

    # ========== Processing ==========

    def _update(self) -> None:
        input: BridgeInput | None = self._input
        if input is None or not input.velocity.allocated:
            return
        self._follow_sizes(input.velocity)

        BlendState.push()
        BlendState.set_blend_mode(BlendState.BlendMode.DISABLED)
        self._bridge(input)
        BlendState.pop()

    def draw(self) -> None:
        """A stage draws nothing; a monitor draws its textures."""
