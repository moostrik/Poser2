"""The fluid stage: the layer that runs the Navier–Stokes simulation on what the bridges give it.

`FluidInput` is what the app hands it each tick: `dt` and one `Injection` per bridge output the
fluid takes. `FluidOutput` is what it hands out. All hold values only: a texture field holds the
`Fbo`, `SwapFbo` or `Tensor` object itself, never a buffer taken from it. The fluid knows
injections, not cameras.

`FluidLayer` is the shape the 2D and the 3D fluid layer share: it injects each injection at its
weight and steps the fluid on the input's `dt`. Its texture is the show, which the composite draws;
the monitors draw its output's `textures`."""

from abc import abstractmethod
from dataclasses import dataclass, field

from modules.gl import Texture, Style, Blit
from modules.settings import Field
from .stage_layer import StageLayer, StageLayerSettings, StageOutput


@dataclass
class Injection:
    """One bridge output as one fluid takes it: `velocity`, `density`, `temperature`, `onset` and
    `spin` from the bridge output, at `weight`, 0 to 1, the injection's share of the fluid's
    velocity, density and temperature, as a layer's weight in a mix. The app sets the weight from
    whatever it reads; the fluid never knows why. `color` is the color the 2D fluid gives the
    injection's density at its output, where the 2D density is an amount; the 3D fluid's density
    arrives colored and ignores it."""
    velocity: Texture
    density: Texture
    temperature: Texture
    onset: Texture | None = None
    spin: Texture | None = None
    weight: float = 1.0
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class FluidInput:
    """What a fluid takes this tick: `dt`, the render tick's, and its injections."""
    dt: float
    injections: list[Injection] = field(default_factory=list)


@dataclass
class FluidOutput(StageOutput):
    """The fluid stage's textures: `density`, the show."""
    density: Texture


class FluidLayerSettings(StageLayerSettings):
    """What the 2D and the 3D fluid layer share; each adds its fluid's group."""
    blend_mode: Field[Style.BlendMode] =    Field(Style.BlendMode.ADD)


class FluidLayer(StageLayer[FluidInput, FluidOutput]):
    """The fluid stage's layer. `update` injects each injection (`_inject`) and steps the fluid
    (`_step`); `texture` is the show."""

    def __init__(self, settings: FluidLayerSettings) -> None:
        super().__init__(settings)
        self.settings: FluidLayerSettings = settings

    @property
    @abstractmethod
    def texture(self) -> Texture:
        """The show: the output's density."""
        ...

    @abstractmethod
    def _inject(self, injection: Injection) -> None:
        """Add the injection's velocity, density and temperature to the fluid at its weight."""

    @abstractmethod
    def _step(self, dt: float) -> None:
        """Advance the fluid one tick."""

    # ========== Processing ==========

    def _update(self) -> None:
        input: FluidInput | None = self._input
        if input is None:
            return

        Style.push_style()
        Style.set_blend_mode(Style.BlendMode.DISABLED)
        for injection in input.injections:
            self._inject(injection)
        self._step(input.dt)
        Style.pop_style()

    # ========== Rendering ==========

    def draw(self) -> None:
        Style.push_style()
        Style.set_blend_mode(self.settings.blend_mode)
        if self.texture.allocated:
            Blit.use(self.texture)
        Style.pop_style()
