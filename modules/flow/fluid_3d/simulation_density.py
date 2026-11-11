"""The density in the simulation's cells: an RGBA16F `GL_TEXTURE_3D` on the simulation's grid, carried by the fluid's advection
and marched front to back into the output (`Composite3D`)."""

from OpenGL.GL import GL_RGBA16F, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT, glMemoryBarrier

from modules.gl import Texture, Texture3D, SwapTexture3D
from ..port import Port, TextureKind
from .density_carrier import CarrierStep, DensityCarrier
from .field_advection import FieldAdvection
from .fluid_settings import Fluid3DSettings
from .shaders import MarchLook, Inject3D, InjectMode, Composite3D, CompositeLight

_BARRIER_IMAGE: int = int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)


class SimulationDensity(DensityCarrier):
    """The SIMULATION carrier: the density a volume on the simulation's grid."""

    def __init__(self, settings: Fluid3DSettings, advection: FieldAdvection) -> None:
        self.settings: Fluid3DSettings = settings
        self._advection: FieldAdvection = advection
        self._density: SwapTexture3D = SwapTexture3D()
        self._scratch: Texture3D = Texture3D()      # the MacCormack predictor
        self._walls: tuple | None = None
        self._z_extent: float = 1.0
        self._inject_shader: Inject3D = Inject3D()
        self._composite_shader: Composite3D = Composite3D()

    def allocate(self, width: int, height: int, layers: int, extent: float, output_width: int, output_height: int) -> None:
        self._density.allocate(width, height, layers, GL_RGBA16F)
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

    def reload(self) -> None:
        self._inject_shader.reload()
        self._composite_shader.reload()

    def reset(self) -> None:
        self._density.clear_all()

    def add(self, texture: Texture, gain: float, dt: float, layer: float, spread: float, mode: InjectMode = InjectMode.RATE,
            target_gain: float = 1.0, z_position: Texture | None = None) -> None:
        if not self.allocated:
            return
        strength: float = gain if mode == InjectMode.FILL else gain * dt
        self._inject_shader.use(texture, self._density, layer, spread, strength, internal_format=GL_RGBA16F, mode=int(mode), gain=target_gain,
                                z_position=z_position)
        glMemoryBarrier(_BARRIER_IMAGE)

    def update(self, step: CarrierStep) -> None:
        if not self.allocated:
            return
        if self._advection.carry(step.advection, self._density, self._scratch, step.velocity, step.obstacle,
                                 step.aspect, step.z_extent, step.step, step.dissipation, GL_RGBA16F,
                                 step.has_obstacles, evaporation=step.evaporation, guard=step.guard, faces=step.faces,
                                 limiter=step.limiter, blend=step.blend):
            self._apply_walls()

    def composite(self, output: Texture, look: MarchLook, light: CompositeLight | None = None) -> None:
        if not self.allocated:
            return
        self._composite_shader.use(self._density, output, look, light=light)
        glMemoryBarrier(_BARRIER_IMAGE)

    @property
    def weight(self) -> Texture3D | None:
        return self._density if self.allocated else None

    @property
    def ports(self) -> dict[str, Port]:
        return {'density': Port(self._density, TextureKind.IMAGE, self._z_extent)} if self.allocated else {}

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
