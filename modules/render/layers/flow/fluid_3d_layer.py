"""The 3D fluid layer: `Fluid3D`, the volumetric Navier–Stokes simulation, on its injections.

The density volume holds the colored density (RGBA16F GL_TEXTURE_3D on the simulation's grid);
`Composite3D` marches it into the RGBA16F 2D output, the layer's texture.
"""

# Local application imports
from modules.gl import Texture
from modules.flow import Fluid3D, Fluid3DSettings, HasPorts
from .fluid_layer import FluidLayer, FluidLayerSettings, FluidOutput, Injection


class Fluid3DLayerSettings(Fluid3DSettings, FluidLayerSettings):
    """Configuration for Fluid3DLayer: the stage's settings and its fluid's own, one panel, the stage's
    first; the stage's width and height are the fluid's simulation grid."""


class Fluid3DLayer(FluidLayer):
    """The 3D fluid layer: each injection's boundary velocity dragged toward and body force acceleration added,
    their z as w, its densities and temperatures into the volume at its weight; one step per tick; the
    composited density as the texture."""

    def __init__(self, settings: Fluid3DLayerSettings) -> None:
        super().__init__(settings)
        self.settings: Fluid3DLayerSettings = settings
        self._fluid: Fluid3D = Fluid3D(settings)

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
        super().deallocate()

    def reset(self) -> None:
        self._fluid.reset()

    # ========== Processing ==========

    def _inject(self, injection: Injection) -> None:
        fluid: Fluid3D = self._fluid
        # The boundary's velocity is dragged toward, the body force's acceleration added × dt; the z (blue) of
        # each is the w
        fluid.drag_velocity(injection.boundary_velocity, injection.weight)
        if injection.body_force_acceleration is not None:
            fluid.add_acceleration(injection.body_force_acceleration, injection.weight)
        fluid.add_boundary_density(injection.boundary_density, injection.weight)
        if injection.body_force_density is not None:
            fluid.add_body_force_density(injection.body_force_density, injection.weight)
        fluid.add_boundary_temperature(injection.boundary_temperature, injection.weight)
        if injection.body_force_temperature is not None:
            fluid.add_body_force_temperature(injection.body_force_temperature, injection.weight)

    def _step(self, dt: float) -> None:
        self._fluid.update(dt)
