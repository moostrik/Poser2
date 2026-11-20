"""The density: an RGBA16F `GL_TEXTURE_3D` of color × amount and the amount, the color in `density.color_space`,
at its own `density.width` × `density.height` × `density.depth` over the simulation's box, by default the dye's
size, carried by the water's
velocity read by uv between the simulation's cells; marched front to back into the output (`Composite3D`) when it is
drawn, else the fine density's color. The buoyancy reads its amount, by uv."""

from OpenGL.GL import GL_RGBA16F, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT, glMemoryBarrier

from modules.gl import Texture, Texture3D, SwapTexture3D
from ..port import ColorSpace, Port, TextureKind
from .density_carrier import CarrierStep, DensityCarrier
from .field_advection import FieldAdvection
from ..density_blend import DensityBlend
from .fluid_settings import Fluid3DSettings
from .shaders import MarchLook, InjectDensity3D, Composite3D, CompositeLight

_BARRIER_IMAGE: int = int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)


class Density(DensityCarrier):
    """The density, always on: color × amount at its own size."""

    def __init__(self, settings: Fluid3DSettings, advection: FieldAdvection) -> None:
        self.settings: Fluid3DSettings = settings
        self._advection: FieldAdvection = advection
        self._density: SwapTexture3D = SwapTexture3D()
        self._scratch: Texture3D = Texture3D()      # the MacCormack predictor
        self._walls: tuple | None = None
        self._z_extent: float = 1.0
        self._inject_shader: InjectDensity3D = InjectDensity3D()
        self._composite_shader: Composite3D = Composite3D()

    def allocate(self, width: int, height: int, layers: int, extent: float, output_width: int, output_height: int) -> None:
        density = self.settings.density
        self._density.allocate(int(density.width), int(density.height), int(density.depth), GL_RGBA16F)
        self._z_extent = extent
        self._inject_shader.allocate()
        self._composite_shader.allocate()
        self._apply_walls()

    def deallocate(self) -> None:
        self._density.deallocate()
        self._scratch.deallocate()
        self._inject_shader.deallocate()
        self._composite_shader.deallocate()

    @property
    def allocated(self) -> bool:
        return self._density.allocated

    def reset(self) -> None:
        self._density.clear_all()

    def add(self, amount: Texture, color: Texture, strength: float, dt: float, spread: float, blend: DensityBlend,
            gain: float, release: float, z_position: Texture | None = None) -> None:
        if not self.allocated:
            return
        self._inject_shader.use(amount, color, self._density, spread, strength, int(blend), gain, release,
                                self._oklab, z_position)
        glMemoryBarrier(_BARRIER_IMAGE)

    def update(self, step: CarrierStep) -> None:
        if not self.allocated:
            return
        if self._advection.carry(step.advection, self._density, self._scratch, step.velocity, step.obstacle,
                                 step.aspect, step.z_extent, step.step, step.dissipation, GL_RGBA16F,
                                 step.has_obstacles, evaporation=step.evaporation,
                                 evaporation_knee=step.evaporation_knee, guard=step.guard,
                                 limiter=step.limiter):
            self._apply_walls()

    def composite(self, output: Texture, look: MarchLook, light: CompositeLight | None = None) -> None:
        if not self.allocated:
            return
        self._composite_shader.use(self._density, output, look, light=light, oklab=self._oklab)
        glMemoryBarrier(_BARRIER_IMAGE)

    @property
    def weight(self) -> Texture3D | None:
        return self._density if self.allocated else None

    @property
    def ports(self) -> dict[str, Port]:
        return {'density': Port(self._density, TextureKind.IMAGE, self._z_extent,
                                color_space=self.settings.density.color_space)} if self.allocated else {}

    @property
    def _oklab(self) -> bool:
        """Whether the color is stored as Oklab."""
        return self.settings.density.color_space == ColorSpace.OKLAB

    def set_walls(self, xy: tuple[int, tuple[float, float, float, float]], z: tuple[int, tuple[float, float, float, float]]) -> None:
        self._walls = (xy, z)
        self._apply_walls()

    def _apply_walls(self) -> None:
        if self._walls is None:
            return
        xy, z = self._walls
        for volume in (self._density, self._scratch):
            volume.set_wrap(*xy)
            volume.set_wrap_z(*z)
