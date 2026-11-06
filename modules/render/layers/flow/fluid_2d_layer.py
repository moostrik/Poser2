"""The 2D fluid layer: `Fluid2D`, the Navier–Stokes simulation on the plane, on its injections.

The density keeps the injections apart: injection i's amount goes into channel i of the RGBA density,
four at most, and the output colors each channel with its injection's color (`DensityColorize`),
the brightness capped at white and the opacity the strongest channel's.
"""

# Standard library imports
import logging

# Third-party imports
from OpenGL.GL import GL_RGBA16F

# Local application imports
from modules.gl import Fbo, Texture
from modules.flow import Fluid2D, Fluid2DSettings
from ...shaders import DensityColorize
from .fluid_layer import FluidLayer, FluidLayerSettings, FluidInput, FluidOutput, Injection
from .port import Port, TextureKind

logger = logging.getLogger(__name__)

CHANNELS: int = 4       # the density's channels, one injection each


class Fluid2DLayerSettings(Fluid2DSettings, FluidLayerSettings):
    """Configuration for Fluid2DLayer: the stage's settings and its fluid's own, one panel, the stage's
    first; the stage's width and height are the fluid's simulation grid."""


class Fluid2DLayer(FluidLayer):
    """The 2D fluid layer: each injection's velocity and temperature into the fluid at its weight, its
    density's amount into the injection's own channel; one step per tick; the channels colored at the
    output, the show."""

    def __init__(self, settings: Fluid2DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Fluid2DLayerSettings = settings
        self._fluid: Fluid2D = Fluid2D(settings)
        self._colors: list[tuple[float, float, float, float]] = [(0.0, 0.0, 0.0, 0.0)] * CHANNELS    # each channel's injection's color
        self._colorized: Fbo = Fbo()
        self._colorize_shader: DensityColorize = DensityColorize()
        self._channel: int = 0      # the channel the next injection's amount goes into
        self._warned_channels: bool = False

    @property
    def texture(self) -> Texture:
        return self._colorized

    @property
    def output(self) -> FluidOutput:
        fluid: Fluid2D = self._fluid
        return FluidOutput(self._colorized, diagnostics={
            'density_channels': Port(fluid.density, TextureKind.IMAGE),
            'velocity': Port(fluid.velocity, TextureKind.VELOCITY),
            'pressure': Port(fluid.pressure, TextureKind.SCALAR),
            'temperature': Port(fluid.temperature, TextureKind.SCALAR),
            'divergence': Port(fluid.divergence, TextureKind.SCALAR),
            'vorticity': Port(fluid.vorticity_curl, TextureKind.SCALAR),
            'buoyancy': Port(fluid.buoyancy, TextureKind.VELOCITY),
            'obstacle': Port(fluid.obstacle, TextureKind.SCALAR),
        })

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        self._fluid.allocate(width, height)
        self._colorized.allocate(width, height, GL_RGBA16F)
        self._colorize_shader.allocate()

    def deallocate(self) -> None:
        self._fluid.deallocate()
        self._colorized.deallocate()
        self._colorize_shader.deallocate()
        super().deallocate()

    def reset(self) -> None:
        self._fluid.reset()

    # ========== Processing ==========

    def _update(self) -> None:
        input: FluidInput | None = self._input
        if input is not None:
            self._colors = [(*injection.color, 1.0) for injection in input.injections[:CHANNELS]]
            self._colors += [(0.0, 0.0, 0.0, 0.0)] * (CHANNELS - len(self._colors))
            if len(input.injections) > CHANNELS and not self._warned_channels:
                self._warned_channels = True
                logger.warning("Fluid2DLayer: %d injections, the density has %d channels; the rest add no density", len(input.injections), CHANNELS)
        super()._update()

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
        self._fluid.update(dt)      # reallocates itself on a size change
        self._colorize_shader.reload()
        self._colorized.begin()
        self._colorize_shader.use(self._fluid.density, self._colors)
        self._colorized.end()
