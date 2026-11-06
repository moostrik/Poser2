"""The density on the grid: an RGBA16F `GL_TEXTURE_3D`, the simulation's size or `density.size` times finer
in x and y, carried by the fluid's advection and marched front to back into the output (`Composite3D`)."""

from OpenGL.GL import GL_RGBA16F, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT, glMemoryBarrier

from modules.gl import Texture, Texture3D, SwapTexture3D
from ..port import Port, TextureKind
from .density_carrier import CarrierStep, DensityCarrier
from .field_advection import FieldAdvection
from .fluid_settings import Fluid3DSettings
from .shaders import Inject3D, Composite3D

_BARRIER_IMAGE: int = int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)


class GridDensity(DensityCarrier):
    """The grid carrier: the density a volume on the simulation's grid, `density.size` times finer in x and y."""

    def __init__(self, settings: Fluid3DSettings, advection: FieldAdvection) -> None:
        self.settings: Fluid3DSettings = settings
        self._advection: FieldAdvection = advection
        self._density: SwapTexture3D = SwapTexture3D()
        self._scratch: Texture3D = Texture3D()      # the MacCormack predictor
        self._walls: tuple | None = None
        self._z_extent: float = 1.0
        self._inject_shader: Inject3D = Inject3D()
        self._composite_shader: Composite3D = Composite3D()

    def allocate(self, width: int, height: int, layers: int, output_width: int, output_height: int) -> None:
        size: int = max(1, self.settings.density.size)
        self._density.allocate(width * size, height * size, layers, GL_RGBA16F)
        self._z_extent = layers / max(1, width)
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

    def add(self, texture: Texture, gain: float, dt: float, layer: float, spread: float) -> None:
        if not self.allocated:
            return
        self._inject_shader.use(texture, self._density, layer, spread, gain * dt, internal_format=GL_RGBA16F)
        glMemoryBarrier(_BARRIER_IMAGE)

    def update(self, step: CarrierStep) -> None:
        if not self.allocated:
            return
        if self._advection.carry(step.advection, self._density, self._scratch, step.velocity, step.obstacle,
                                 step.aspect, step.z_extent, step.step, step.dissipation, GL_RGBA16F,
                                 step.has_obstacles, evaporation=step.evaporation):
            self._apply_walls()

    def composite(self, output: Texture) -> None:
        if not self.allocated:
            return
        volume = self.settings.volume
        self._composite_shader.use(self._density, output, emission=volume.emission, absorption=volume.absorption,
                                   tint=volume.tint, fog=volume.fog, glow=volume.glow)
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
