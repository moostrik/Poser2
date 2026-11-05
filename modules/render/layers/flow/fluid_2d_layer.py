"""The 2D fluid layer: `Fluid2D`, the Navier–Stokes simulation on the plane, on its injections."""

# Standard library imports
from enum import IntEnum, auto

# Local application imports
from modules.gl import Texture
from modules.settings import Field, Group
from modules.flow import Visualizer, VisualizationFieldSettings, Fluid2D, Fluid2DSettings
from modules.utils import HotReloadMethods
from .fluid_layer import FluidLayer, FluidLayerSettings, Injection


class Fluid2DDrawMode(IntEnum):
    """Draw modes for Fluid2DLayer, the fluid's fields.

    VELOCITY - Advected velocity field (RG16F)
    DENSITY - The density, with its color (RGBA16F)
    PRESSURE - Pressure field (R32F)
    TEMPERATURE - Temperature field (R32F)
    DIVERGENCE - Velocity divergence (R32F)
    VORTICITY - Vorticity curl (R32F)
    BUOYANCY - Buoyancy force (RG32F)
    OBSTACLE - Obstacle mask (R8)
    """
    VELOCITY = 0
    DENSITY = auto()
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
    """The 2D fluid layer: each injection's velocity, density and temperature into the fluid at its
    weight; one step per tick; the field `draw_mode` names through the visualizer as the texture."""

    def __init__(self, settings: Fluid2DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Fluid2DLayerSettings = settings
        self._fluid: Fluid2D = Fluid2D(settings.fluid)
        self._visualizer: Visualizer = Visualizer(settings.visualization)

        self._hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self._visualizer.texture

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        self._fluid.allocate(width, height)
        self._visualizer.allocate(self._fluid.sim_width, self._fluid.sim_height)

    def deallocate(self) -> None:
        self._fluid.deallocate()
        self._visualizer.deallocate()

    def reset(self) -> None:
        self._fluid.reset()

    # ========== Processing ==========

    def _inject(self, injection: Injection) -> None:
        self._fluid.add_velocity(injection.velocity, injection.weight)
        self._fluid.add_density(injection.density, injection.weight)
        self._fluid.add_temperature(injection.temperature, injection.weight)

    def _step(self, dt: float) -> None:
        size: tuple[int, int] = (self._fluid.sim_width, self._fluid.sim_height)
        self._fluid.update(dt)      # reallocates itself on a size change
        if (self._fluid.sim_width, self._fluid.sim_height) != size:
            self._visualizer.allocate(self._fluid.sim_width, self._fluid.sim_height)

    def _visualize(self) -> None:
        fluid: Fluid2D = self._fluid
        textures: dict[Fluid2DDrawMode, Texture] = {
            Fluid2DDrawMode.VELOCITY: fluid.velocity,
            Fluid2DDrawMode.DENSITY: fluid.density,
            Fluid2DDrawMode.PRESSURE: fluid.pressure,
            Fluid2DDrawMode.TEMPERATURE: fluid.temperature,
            Fluid2DDrawMode.DIVERGENCE: fluid.divergence,
            Fluid2DDrawMode.VORTICITY: fluid.vorticity_curl,
            Fluid2DDrawMode.BUOYANCY: fluid.buoyancy,
            Fluid2DDrawMode.OBSTACLE: fluid.obstacle,
        }
        self._visualizer.process(textures.get(self.settings.draw_mode, fluid.density))
