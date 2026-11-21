"""The 3D fluid layer: `Fluid3D`, the volumetric Navier–Stokes simulation, on its injections.

The density volume holds the colored density (RGBA16F GL_TEXTURE_3D on the simulation's grid);
`Composite3D` marches it into the RGBA16F 2D output, the layer's texture.
"""

# Local application imports
from modules.gl import GpuTimer, BlendState, Texture
from modules.flow import CompositeSettings, Fluid3D, Fluid3DSettings, HasPorts
from modules.settings import Field, Group
from ...general.lut_grade import LutGrade, LutSettings
from .fluid_layer import FluidLayer, FluidLayerSettings, FluidOutput, Injection
from .monitor_2d_layer import DEBUG_TINT, MAIN_TINT
from .monitor_3d_layer import Monitor3DLayerSettings


class Fluid3DCompositeSettings(CompositeSettings):
    """The fluid's composite, how the layer draws its picture over the layers before it, and the LUT it is drawn
    through."""
    blend_mode: Field[BlendState.BlendMode] =    Field(BlendState.BlendMode.ADD, description="How the show is drawn over the layers before it")
    lut: Group[LutSettings] =               Group(LutSettings)


class Fluid3DLayerSettings(Fluid3DSettings, FluidLayerSettings):
    """Configuration for Fluid3DLayer: the stage's settings and its fluid's own, one panel, the stage's
    first; the fluid's width, height and depth are the stage's working size."""
    composite: Group[Fluid3DCompositeSettings] = Group(Fluid3DCompositeSettings)

    monitor_main:  Group[Monitor3DLayerSettings] = Group(Monitor3DLayerSettings, last=True, tint=MAIN_TINT)
    monitor_debug: Group[Monitor3DLayerSettings] = Group(Monitor3DLayerSettings, last=True, tint=DEBUG_TINT)


class Fluid3DLayer(FluidLayer):
    """The 3D fluid layer: each injection's boundary velocity dragged toward and body force acceleration added,
    their z as w, its density and temperature into the volume at its weight; one step per tick; the
    composited density as the texture."""

    def __init__(self, settings: Fluid3DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Fluid3DLayerSettings = settings
        self._fluid: Fluid3D = Fluid3D(settings)
        self._inject_timer: GpuTimer = GpuTimer("GPU fluid inject")     # an injection's drag, force and sources
        self._grade: LutGrade = LutGrade(settings.composite.lut)

    @property
    def blend_mode(self) -> BlendState.BlendMode:
        """The panel's blend for a premultiplied picture: the composite's color is already times its coverage."""
        return self.settings.composite.blend_mode.premultiplied

    @property
    def texture(self) -> Texture:
        """Composited 2D density output (RGBA16F, color baked in)."""
        return self._fluid.density

    @property
    def output(self) -> FluidOutput:
        return FluidOutput(self._fluid.density)

    def _components(self) -> dict[str, HasPorts]:
        return {'': self._fluid}

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
        fluid: Fluid3D = self._fluid
        self._inject_timer.begin()
        self._inject_into(fluid, injection)
        self._inject_timer.end()

    @staticmethod
    def _inject_into(fluid: Fluid3D, injection: Injection) -> None:
        # The boundary's velocity is dragged toward, the body force's acceleration added × dt; the z (blue) of
        # each is the w. Each goes in at its own z position
        fluid.drag_velocity(injection.velocity, injection.weight, injection.velocity_z_position)
        if injection.acceleration is not None:
            fluid.add_acceleration(injection.acceleration, injection.weight, injection.acceleration_z_position)
        fluid.add_density(injection.density, injection.color, injection.weight, injection.density_z_position)
        fluid.add_temperature(injection.temperature, injection.weight, injection.temperature_z_position)
        if injection.expansion is not None:
            fluid.add_expansion(injection.expansion, injection.weight, injection.expansion_z_position)

    def _step(self, dt: float) -> None:
        self._fluid.update(dt)
