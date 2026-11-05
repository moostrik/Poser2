"""The 2D fluid layer: `Fluid2D`, the Navier–Stokes simulation on the plane, on its injections.

The density keeps the injections apart: injection i's amount goes into channel i of the RGBA density,
four at most, and the output colors each channel with its injection's color (`DensityColorize`),
the brightness capped at white and the opacity the strongest channel's.
"""

# Standard library imports
from enum import IntEnum, auto
import logging

# Third-party imports
from OpenGL.GL import GL_RGBA16F

# Local application imports
from modules.gl import Fbo, Texture
from modules.settings import Field, Group
from modules.flow import Visualizer, VisualizationFieldSettings, Fluid2D, Fluid2DSettings
from modules.utils import HotReloadMethods
from ...shaders import DensityColorize
from .fluid_layer import FluidLayer, FluidLayerSettings, FluidInput, Injection

logger = logging.getLogger(__name__)

CHANNELS: int = 4       # the density's channels, one injection each


class Fluid2DDrawMode(IntEnum):
    """Draw modes for Fluid2DLayer, the fluid's fields.

    VELOCITY - Advected velocity field (RG16F)
    DENSITY - The density, each injection's channel in its color (RGBA16F)
    DENSITY_RAW - The density's channels before the colors (RGBA16F)
    PRESSURE - Pressure field (R32F)
    TEMPERATURE - Temperature field (R32F)
    DIVERGENCE - Velocity divergence (R32F)
    VORTICITY - Vorticity curl (R32F)
    BUOYANCY - Buoyancy force (RG32F)
    OBSTACLE - Obstacle mask (R8)
    """
    VELOCITY = 0
    DENSITY = auto()
    DENSITY_RAW = auto()
    PRESSURE = auto()
    TEMPERATURE = auto()
    DIVERGENCE = auto()
    VORTICITY = auto()
    BUOYANCY = auto()
    OBSTACLE = auto()


class Fluid2DLayerSettings(FluidLayerSettings):
    """Configuration for Fluid2DLayer."""
    draw_mode: Field[Fluid2DDrawMode] =     Field(Fluid2DDrawMode.DENSITY)

    fluid: Group[Fluid2DSettings]                    = Group(Fluid2DSettings)
    visualization: Group[VisualizationFieldSettings] = Group(VisualizationFieldSettings)


class Fluid2DLayer(FluidLayer):
    """The 2D fluid layer: each injection's velocity and temperature into the fluid at its weight, its
    density's amount into the injection's own channel; one step per tick; the channels colored at the
    output; the field `draw_mode` names through the visualizer as the texture."""

    def __init__(self, settings: Fluid2DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Fluid2DLayerSettings = settings
        self._fluid: Fluid2D = Fluid2D(settings.fluid)
        self._visualizer: Visualizer = Visualizer(settings.visualization)
        self._colors: list[tuple[float, float, float, float]] = [(0.0, 0.0, 0.0, 0.0)] * CHANNELS    # each channel's injection's color
        self._colorized: Fbo = Fbo()
        self._colorize_shader: DensityColorize = DensityColorize()
        self._channel: int = 0      # the channel the next injection's amount goes into
        self._warned_channels: bool = False

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self._visualizer.texture

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        self._fluid.allocate(width, height)
        self._visualizer.allocate(self._fluid.sim_width, self._fluid.sim_height)
        self._colorized.allocate(width, height, GL_RGBA16F)
        self._colorize_shader.allocate()

    def deallocate(self) -> None:
        self._fluid.deallocate()
        self._visualizer.deallocate()
        self._colorized.deallocate()
        self._colorize_shader.deallocate()

    def reset(self) -> None:
        self._fluid.reset()

    # ========== Processing ==========

    def update(self) -> None:
        input: FluidInput | None = self._input
        if input is not None:
            self._colors = [(*injection.color, 1.0) for injection in input.injections[:CHANNELS]]
            self._colors += [(0.0, 0.0, 0.0, 0.0)] * (CHANNELS - len(self._colors))
            if len(input.injections) > CHANNELS and not self._warned_channels:
                self._warned_channels = True
                logger.warning("Fluid2DLayer: %d injections, the density has %d channels; the rest add no density", len(input.injections), CHANNELS)
        super().update()

    def _inject(self, injection: Injection) -> None:
        fluid: Fluid2D = self._fluid
        fluid.add_velocity(injection.velocity, injection.weight)
        channel: int = self._channel
        if channel < CHANNELS:
            fluid.add_density_channel(injection.density, channel, injection.weight)
        fluid.add_temperature(injection.temperature, injection.weight)
        self._channel = channel + 1

    def _step(self, dt: float) -> None:
        self._channel = 0
        size: tuple[int, int] = (self._fluid.sim_width, self._fluid.sim_height)
        self._fluid.update(dt)      # reallocates itself on a size change
        if (self._fluid.sim_width, self._fluid.sim_height) != size:
            self._visualizer.allocate(self._fluid.sim_width, self._fluid.sim_height)

    def _visualize(self) -> None:
        fluid: Fluid2D = self._fluid
        self._colorize_shader.reload()
        self._colorized.begin()
        self._colorize_shader.use(fluid.density, self._colors)
        self._colorized.end()
        textures: dict[Fluid2DDrawMode, Texture] = {
            Fluid2DDrawMode.VELOCITY: fluid.velocity,
            Fluid2DDrawMode.DENSITY: self._colorized,
            Fluid2DDrawMode.DENSITY_RAW: fluid.density,
            Fluid2DDrawMode.PRESSURE: fluid.pressure,
            Fluid2DDrawMode.TEMPERATURE: fluid.temperature,
            Fluid2DDrawMode.DIVERGENCE: fluid.divergence,
            Fluid2DDrawMode.VORTICITY: fluid.vorticity_curl,
            Fluid2DDrawMode.BUOYANCY: fluid.buoyancy,
            Fluid2DDrawMode.OBSTACLE: fluid.obstacle,
        }
        self._visualizer.process(textures.get(self.settings.draw_mode, self._colorized))
