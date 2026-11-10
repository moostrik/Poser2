"""The density upscaled past the simulation: its amount a one-channel `GL_TEXTURE_3D` of `upscaled.width` ×
`upscaled.height` × `upscaled.depth` over the simulation's box, its color the simulation density's, which the
fluid keeps on its own cells and this borrows.

The color never holds more detail than the bridge's density it comes from, so it stays on the grid; the amount,
moved by the grid's velocity read between its cells, is stretched into threads finer than a cell and keeps them
from step to step. A density layer that spans several of the grid's layers moves with the velocity at its middle
or averaged over them (`upscaled.sample`). The composite marches the amount's layers, the color the color
volume's color × amount over its amount at the same place."""

import numpy as np
from OpenGL.GL import (GL_CLAMP_TO_BORDER, GL_FLOAT, GL_R16F, GL_REPEAT, GL_RGBA, GL_RGBA16F, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT,
                       GL_TEXTURE_3D, GL_TEXTURE_FETCH_BARRIER_BIT, glBindTexture, glMemoryBarrier, glTexSubImage3D)

from modules.gl import Texture, Texture3D, SwapTexture3D
from ..port import Port, TextureKind
from .density_carrier import CarrierStep, DensityCarrier
from .field_advection import FieldAdvection
from .fluid_settings import DensitySample, Fluid3DSettings
from .simulation_density import SimulationDensity
from .shaders import MarchLook, Composite3D, CompositeLight, InjectAmount3D, InjectMode, NoiseCoordinates3D, TurbulentAdvect3D, VelocityAverage3D
from .swirl_noise import swirl_noise

_BARRIER: int = int(GL_TEXTURE_FETCH_BARRIER_BIT) | int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

NOISE_SIZE: int = 64            # the swirl tile's texels a side
NOISE_BAND: tuple[float, float] = (8.0, 16.0)   # its wavelengths in texels
NOISE_CELLS: float = 8.0        # the grid's cells a tile spans: the swirls' wavelengths 1 to 2 cells
NOISE_CYCLE: float = 1.0        # s of the fluid's time between two resets of one displacement


class UpscaledDensity(DensityCarrier):
    """The upscaled density: the amount at `upscaled`'s size, the color the simulation density's."""

    def __init__(self, settings: Fluid3DSettings, advection: FieldAdvection, color: SimulationDensity) -> None:
        self.settings: Fluid3DSettings = settings
        self._advection: FieldAdvection = advection
        self._color: SimulationDensity = color      # color × amount and the amount, the fluid's, moved by it
        self._amount: SwapTexture3D = SwapTexture3D()
        self._scratch: Texture3D = Texture3D()      # the MacCormack predictor
        self._average: Texture3D = Texture3D(wrap=GL_CLAMP_TO_BORDER, border_color=(0.0, 0.0, 0.0, 0.0))
        self._walls: tuple | None = None
        self._z_extent: float = 1.0
        self._inject_shader: InjectAmount3D = InjectAmount3D()
        self._average_shader: VelocityAverage3D = VelocityAverage3D()
        self._composite_shader: Composite3D = Composite3D()
        # The turbulence: the swirl tile, two displacements of the water half a cycle apart, the clock
        self._noise: Texture3D = Texture3D(wrap=GL_REPEAT)
        self._displacements: tuple[SwapTexture3D, SwapTexture3D] = (SwapTexture3D(), SwapTexture3D())
        self._coordinates_shader: NoiseCoordinates3D = NoiseCoordinates3D()
        self._turbulent_shader: TurbulentAdvect3D = TurbulentAdvect3D()
        self._clock: float = 0.0

    def allocate(self, width: int, height: int, layers: int, extent: float, output_width: int, output_height: int) -> None:
        upscaled = self.settings.upscaled
        density_layers: int = int(upscaled.depth)
        self._amount.allocate(int(upscaled.width), int(upscaled.height), density_layers, GL_R16F)
        self._z_extent = extent
        if layers > density_layers:
            self._average.allocate(width, height, density_layers, GL_RGBA16F)
        for displacement in self._displacements:
            displacement.allocate(width, height, layers, GL_RGBA16F)
            displacement.clear_all()
        tile = np.concatenate((swirl_noise(NOISE_SIZE, *NOISE_BAND), np.zeros((NOISE_SIZE,) * 3 + (1,), np.float32)), axis=-1)
        self._noise.allocate(NOISE_SIZE, NOISE_SIZE, NOISE_SIZE, GL_RGBA16F)
        glBindTexture(GL_TEXTURE_3D, self._noise.tex_id)
        glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, NOISE_SIZE, NOISE_SIZE, NOISE_SIZE, GL_RGBA, GL_FLOAT, np.ascontiguousarray(tile))
        glBindTexture(GL_TEXTURE_3D, 0)
        self._clock = 0.0
        for shader in (self._inject_shader, self._average_shader, self._composite_shader, self._coordinates_shader,
                       self._turbulent_shader):
            shader.allocate()
        self._apply_walls()

    def deallocate(self) -> None:
        for volume in (self._amount, self._scratch, self._average, self._noise, *self._displacements):
            volume.deallocate()
        for shader in (self._inject_shader, self._average_shader, self._composite_shader, self._coordinates_shader,
                       self._turbulent_shader):
            shader.deallocate()

    @property
    def allocated(self) -> bool:
        return self._amount.allocated

    def reload(self) -> None:
        for shader in (self._inject_shader, self._average_shader, self._composite_shader, self._coordinates_shader,
                       self._turbulent_shader):
            shader.reload()

    def reset(self) -> None:
        self._amount.clear_all()

    def add(self, texture: Texture, gain: float, dt: float, layer: float, spread: float, mode: InjectMode = InjectMode.RATE,
            target_gain: float = 1.0) -> None:
        if not self.allocated:
            return
        strength: float = gain if mode == InjectMode.FILL else gain * dt
        self._inject_shader.use(texture, self._amount, layer, spread, strength, cubic=self.settings.upscaled.cubic, mode=int(mode), gain=target_gain)
        glMemoryBarrier(_BARRIER)

    def update(self, step: CarrierStep) -> None:
        if not self.allocated:
            return
        velocity: Texture3D = step.velocity
        if self._average.allocated and self.settings.upscaled.sample == DensitySample.AVERAGE:
            self._average_shader.use(step.velocity, self._average)
            glMemoryBarrier(_BARRIER)
            velocity = self._average
        turbulence: float = float(self.settings.upscaled.turbulence)
        if turbulence > 0.0 and step.curl is not None:
            self._turbulent_step(step, velocity, turbulence)
        elif self._advection.carry(step.advection, self._amount, self._scratch, velocity, step.obstacle,
                                   step.aspect, step.z_extent, step.step, step.dissipation, GL_R16F,
                                   step.has_obstacles, evaporation=step.evaporation, measure=2):
            self._apply_walls()

    def _turbulent_step(self, step: CarrierStep, velocity: Texture3D, turbulence: float) -> None:
        """The amount moved semi-Lagrangian by the velocity plus the swirls below the grid. The two displacements
        are carried on the grid each step, and each is reset when its weight in the cross-fade is 0."""
        assert step.curl is not None
        rdx = (1.0, step.aspect, 1.0 / step.z_extent if step.z_extent > 0.0 else 1.0)
        before: float = self._clock / NOISE_CYCLE
        self._clock += step.dt
        after: float = self._clock / NOISE_CYCLE
        for index, displacement in enumerate(self._displacements):
            offset: float = 0.5 * index         # B half a cycle after A
            reset: bool = np.floor(after - offset) > np.floor(before - offset)
            displacement.swap()
            self._coordinates_shader.use(step.velocity, displacement.back, displacement, step.step, rdx, reset)
        glMemoryBarrier(_BARRIER)
        phase = (after % 1.0, (after - 0.5) % 1.0)
        weights = tuple(1.0 - abs(2.0 * p - 1.0) for p in phase)       # 0 at its reset, 1 half a cycle on
        self._amount.swap()
        self._turbulent_shader.use(self._amount.back, self._amount, velocity, step.obstacle, step.curl, self._noise,
                                   self._displacements[0], self._displacements[1], timestep=step.step, rdx=rdx,
                                   extent=(1.0, 1.0 / step.aspect, step.z_extent), dissipation=step.dissipation,
                                   evaporation=step.evaporation, turbulence=turbulence, cell=step.cell,
                                   tiles=1.0 / max(NOISE_CELLS * step.cell, 1e-6), weights=(weights[0], weights[1]),
                                   has_obstacles=step.has_obstacles)
        glMemoryBarrier(_BARRIER)

    def composite(self, output: Texture, look: MarchLook, light: CompositeLight | None = None) -> None:
        if not self.allocated:
            return
        self._composite_shader.use(self._amount, output, look, color=self._color.weight, light=light)
        glMemoryBarrier(_BARRIER)

    @property
    def weight(self) -> Texture3D | None:
        return None         # the buoyancy weighs the simulation density

    @property
    def ports(self) -> dict[str, Port]:
        return {'upscaled_density': Port(self._amount, TextureKind.SCALAR, self._z_extent)} if self.allocated else {}

    def set_walls(self, xy: tuple[int, tuple[float, float, float, float]], z: tuple[int, tuple[float, float, float, float]]) -> None:
        self._walls = (xy, z)
        self._apply_walls()

    def _apply_walls(self) -> None:
        if self._walls is None:
            return
        xy, z = self._walls
        for volume in (self._amount, self._scratch):
            volume.set_wrap(*xy)
            volume.set_wrap_z(*z)
