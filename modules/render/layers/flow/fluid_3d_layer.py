"""The 3D fluid layer: `Fluid3D`, the volumetric Navier–Stokes simulation, on its injections.

The density volume holds the colored density (RGBA16F GL_TEXTURE_3D on the simulation's grid);
`Composite3D` marches it into the RGBA16F 2D output, the layer's texture.
"""

# Local application imports
from modules.gl import GpuTimer, Style, Texture
from modules.flow import CompositeSettings, Fluid3D, Fluid3DSettings, HasPorts
from modules.settings import Field, Group
from .fluid_layer import FluidLayer, FluidLayerSettings, FluidOutput, Injection


class Fluid3DCompositeSettings(CompositeSettings):
    """The fluid's composite, and how the layer draws its picture over the layers before it."""
    blend_mode: Field[Style.BlendMode] =    Field(Style.BlendMode.ADD, description="How the show is drawn over the layers before it")


class Fluid3DLayerSettings(Fluid3DSettings, FluidLayerSettings):
    """Configuration for Fluid3DLayer: the stage's settings and its fluid's own, one panel, the stage's
    first; the fluid's width, height and depth are the stage's working size."""
    composite: Group[Fluid3DCompositeSettings] = Group(Fluid3DCompositeSettings)


class Fluid3DLayer(FluidLayer):
    """The 3D fluid layer: each injection's boundary velocity dragged toward and body force acceleration added,
    their z as w, its densities and temperatures into the volume at its weight; one step per tick; the
    composited density as the texture."""

    def __init__(self, settings: Fluid3DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Fluid3DLayerSettings = settings
        self._fluid: Fluid3D = Fluid3D(settings)
        self._inject_timer: GpuTimer = GpuTimer("GPU fluid inject")     # an injection's drag, force and sources

    @property
    def blend_mode(self) -> Style.BlendMode:
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

    def deallocate(self) -> None:
        self._fluid.deallocate()
        self._inject_timer.deallocate()
        super().deallocate()

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
        fluid.drag_velocity(injection.boundary_velocity, injection.weight, injection.boundary_velocity_z_position)
        if injection.body_force_acceleration is not None:
            fluid.add_acceleration(injection.body_force_acceleration, injection.weight, injection.body_force_acceleration_z_position)
        fluid.add_boundary_density(injection.boundary_density, injection.weight, injection.boundary_density_z_position)
        if injection.body_force_density is not None:
            fluid.add_body_force_density(injection.body_force_density, injection.weight, injection.body_force_density_z_position)
        fluid.add_boundary_temperature(injection.boundary_temperature, injection.weight, injection.boundary_temperature_z_position)
        if injection.body_force_temperature is not None:
            fluid.add_body_force_temperature(injection.body_force_temperature, injection.weight, injection.body_force_temperature_z_position)

    def _step(self, dt: float) -> None:
        self._fluid.update(dt)
