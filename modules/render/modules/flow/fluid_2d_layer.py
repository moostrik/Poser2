"""The 2D fluid layer: `Fluid2D`, the Navier–Stokes simulation on the plane, on its injections.

Each injection goes in as the 3D fluid takes it: the boundary's velocity dragged toward, the body force added × dt,
the density with its color by the blend, the temperature dragged toward, the expansion into the projection. The
density is held as its color times its coverage, and drawn, its picture, through the LUT at the panel's blend.
"""

# Local application imports
from modules.gl import GpuTimer, BlendState, Texture
from modules.settings import Field, Group
from modules.flow import Fluid2D, Fluid2DSettings, HasPorts
from ...general.lut_grade import LutGrade, LutSettings
from .fluid_layer import FluidLayer, FluidLayerSettings, FluidOutput, Injection
from .monitor_2d_layer import DEBUG_TINT, MAIN_TINT, Monitor2DLayerSettings


class Fluid2DLayerSettings(Fluid2DSettings, FluidLayerSettings):
    """Configuration for Fluid2DLayer: the stage's settings and its fluid's own, one panel, the stage's first; the
    stage's width and height are the fluid's simulation grid; how the picture is drawn over the layers before it, and
    the LUT it is drawn through."""
    blend_mode: Field[BlendState.BlendMode] =    Field(BlendState.BlendMode.ADD, newline=True, description="How the show is drawn over the layers before it")
    lut: Group[LutSettings] =               Group(LutSettings, description="The LUT the fluid's picture is drawn through")

    monitor_main:  Group[Monitor2DLayerSettings] = Group(Monitor2DLayerSettings, last=True, tint=MAIN_TINT, description="The fluid stage's monitor in the main panel")
    monitor_debug: Group[Monitor2DLayerSettings] = Group(Monitor2DLayerSettings, last=True, tint=DEBUG_TINT, description="The fluid stage's monitor in the debug panel")


class Fluid2DLayer(FluidLayer):
    """The 2D fluid layer: each injection's boundary velocity dragged toward and body force added, its density and
    temperature at its weight, its expansion into the projection; one step per tick; the density's picture the
    texture."""

    def __init__(self, settings: Fluid2DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Fluid2DLayerSettings = settings
        self._fluid: Fluid2D = Fluid2D(settings)
        self._inject_timer: GpuTimer = GpuTimer("GPU fluid inject")     # an injection's drag, force and sources
        self._grade: LutGrade = LutGrade(settings.lut)

    @property
    def blend_mode(self) -> BlendState.BlendMode:
        """The panel's blend for a premultiplied picture: the density is its color times its coverage."""
        return self.settings.blend_mode.premultiplied

    @property
    def texture(self) -> Texture:
        """The density's picture, RGB premultiplied (RGBA16F)."""
        return self._fluid.picture

    @property
    def output(self) -> FluidOutput:
        return FluidOutput(self._fluid.picture)

    def _components(self) -> dict[str, HasPorts]:
        return {'': self._fluid}    # its fields carry their names: velocity, density, temperature, pressure

    # ========== Lifecycle ==========

    def allocate(self, width: int, height: int, internal_format: int | None = None) -> None:
        self._fluid.allocate(width, height)
        self._grade.allocate()

    def deallocate(self) -> None:
        self._fluid.deallocate()
        self._inject_timer.deallocate()
        self._grade.deallocate()
        super().deallocate()

    def draw(self) -> None:
        """The show, through its LUT where one is picked."""
        BlendState.push()
        BlendState.set_blend_mode(self.blend_mode)
        if self.texture.allocated:
            self._grade.draw(self.texture)
        BlendState.pop()

    def reset(self) -> None:
        self._fluid.reset()

    # ========== Processing ==========

    def _inject(self, injection: Injection) -> None:
        fluid: Fluid2D = self._fluid
        self._inject_timer.begin()
        fluid.drag_velocity(injection.velocity, injection.weight)
        if injection.acceleration is not None:
            fluid.add_acceleration(injection.acceleration, injection.weight)
        fluid.add_density(injection.density, injection.color, injection.weight)
        fluid.add_temperature(injection.temperature, injection.weight)
        if injection.expansion is not None:
            fluid.add_expansion(injection.expansion, injection.weight)
        self._inject_timer.end()

    def _step(self, dt: float) -> None:
        self._fluid.update(dt)      # reallocates itself on a size change
