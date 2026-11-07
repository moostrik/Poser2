"""The density at the output's resolution: its amount a one-channel `GL_TEXTURE_3D` at the output's x and y and
`density.layers` in z, over the grid's depth, its color the grid carrier's volume on the simulation's grid.

The color never holds more detail than the bridge's density it comes from, so it stays on the grid; the amount,
moved by the grid's velocity read between its cells, is stretched into threads finer than a cell and keeps them
from step to step. A density layer that spans several of the grid's layers moves with the velocity at its middle
or averaged over them (`density.sample`). The composite marches the amount's layers, the color the color
volume's color × amount over its amount at the same place."""

from OpenGL.GL import GL_CLAMP_TO_BORDER, GL_R16F, GL_RGBA16F, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT, GL_TEXTURE_FETCH_BARRIER_BIT, glMemoryBarrier

from modules.gl import Texture, Texture3D, SwapTexture3D
from ..port import Port, TextureKind
from .density_carrier import CarrierStep, DensityCarrier
from .field_advection import FieldAdvection
from .fluid_settings import DensitySample, Fluid3DSettings
from .grid_density import GridDensity
from .shaders import Composite3D, InjectAmount3D, VelocityAverage3D

_BARRIER: int = int(GL_TEXTURE_FETCH_BARRIER_BIT) | int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)


class OutputDensity(DensityCarrier):
    """The OUTPUT carrier: the amount at the output's resolution × `density.layers`, the color on the grid."""

    def __init__(self, settings: Fluid3DSettings, advection: FieldAdvection) -> None:
        self.settings: Fluid3DSettings = settings
        self._advection: FieldAdvection = advection
        self._color: GridDensity = GridDensity(settings, advection)     # color × amount and the amount, on the grid
        self._amount: SwapTexture3D = SwapTexture3D()
        self._scratch: Texture3D = Texture3D()      # the MacCormack predictor
        self._average: Texture3D = Texture3D(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        self._walls: tuple | None = None
        self._z_extent: float = 1.0
        self._inject_shader: InjectAmount3D = InjectAmount3D()
        self._average_shader: VelocityAverage3D = VelocityAverage3D()
        self._composite_shader: Composite3D = Composite3D()

    def allocate(self, width: int, height: int, layers: int, output_width: int, output_height: int) -> None:
        density_layers: int = int(self.settings.density.layers)
        self._color.allocate(width, height, layers, output_width, output_height)
        self._amount.allocate(output_width, output_height, density_layers, GL_R16F)
        self._z_extent = layers / max(1, width)
        if layers > density_layers:
            self._average.allocate(width, height, density_layers, GL_RGBA16F)
        for shader in (self._inject_shader, self._average_shader, self._composite_shader):
            shader.allocate()
        self._apply_walls()

    def deallocate(self) -> None:
        self._color.deallocate()
        for volume in (self._amount, self._scratch, self._average):
            volume.deallocate()
        for shader in (self._inject_shader, self._average_shader, self._composite_shader):
            shader.deallocate()

    @property
    def allocated(self) -> bool:
        return self._amount.allocated

    def reload(self) -> None:
        self._color.reload()
        for shader in (self._inject_shader, self._average_shader, self._composite_shader):
            shader.reload()

    def reset(self) -> None:
        self._color.reset()
        self._amount.clear_all()

    def add(self, texture: Texture, gain: float, dt: float, layer: float, spread: float) -> None:
        if not self.allocated:
            return
        self._color.add(texture, gain, dt, layer, spread)
        self._inject_shader.use(texture, self._amount, layer, spread, gain * dt)
        glMemoryBarrier(_BARRIER)

    def update(self, step: CarrierStep) -> None:
        if not self.allocated:
            return
        self._color.update(step)
        velocity: Texture3D = step.velocity
        if self._average.allocated and self.settings.density.sample == DensitySample.AVERAGE:
            self._average_shader.use(step.velocity, self._average)
            glMemoryBarrier(_BARRIER)
            velocity = self._average
        if self._advection.carry(step.advection, self._amount, self._scratch, velocity, step.obstacle,
                                 step.aspect, step.z_extent, step.step, step.dissipation, GL_R16F,
                                 step.has_obstacles, evaporation=step.evaporation):
            self._apply_walls()

    def composite(self, output: Texture) -> None:
        if not self.allocated:
            return
        volume = self.settings.volume
        self._composite_shader.use(self._amount, output, emission=volume.emission, absorption=volume.absorption,
                                   tint=volume.tint, fog=volume.fog, glow=volume.glow, color=self._color.weight)
        glMemoryBarrier(_BARRIER)

    @property
    def weight(self) -> Texture3D | None:
        return self._color.weight

    @property
    def ports(self) -> dict[str, Port]:
        return {'density': Port(self._amount, TextureKind.SCALAR, self._z_extent)} if self.allocated else {}

    def set_walls(self, xy: tuple[int, tuple[float, float, float, float]], z: tuple[int, tuple[float, float, float, float]]) -> None:
        self._walls = (xy, z)
        self._color.set_walls(xy, z)
        self._apply_walls()

    def _apply_walls(self) -> None:
        if self._walls is None:
            return
        xy, z = self._walls
        for volume in (self._amount, self._scratch):
            volume.set_wrap(*xy)
            volume.set_wrap_z(*z)
