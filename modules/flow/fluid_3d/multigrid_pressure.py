"""The pressure by a geometric multigrid: V-cycles over the grid halved down to a few cells deep, red-black
Gauss–Seidel smoothing each level, warm from the last step's pressure (McAdams, Sifakis, Teran 2010)."""

from OpenGL.GL import GL_R32F, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT, GL_TEXTURE_FETCH_BARRIER_BIT, glMemoryBarrier

from modules.gl import Texture3D
from .shaders import SorPressure3D, PressureResidual3D, Restrict3D, Prolong3D

_BARRIER: int = int(GL_TEXTURE_FETCH_BARRIER_BIT) | int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

SMOOTHING: int = 2          # red-black Gauss–Seidel sweeps before and after the coarser level, on each level
COARSEST_SWEEPS: int = 16   # sweeps on the coarsest level, which stands for its solve
SMALLEST: int = 4           # a level is halved while every axis stays at least this many cells


class MultigridPressure:
    """V-cycles on the pressure in place: on each level smooth, the residual onto the next coarser level as its
    right-hand side, the coarser level's correction solved from 0 and added back, smooth again. The walls are
    the solvers' on every level; obstacles are kept on the finest level's smoothing only."""

    def __init__(self) -> None:
        self._smoother: SorPressure3D = SorPressure3D()
        self._residual_shader: PressureResidual3D = PressureResidual3D()
        self._restrict_shader: Restrict3D = Restrict3D()
        self._prolong_shader: Prolong3D = Prolong3D()
        self._residuals: list[Texture3D] = []       # each level's residual but the coarsest's
        self._corrections: list[Texture3D] = []     # each coarser level's correction
        self._sources: list[Texture3D] = []         # each coarser level's right-hand side, the finer residual restricted

    def allocate(self, width: int, height: int, depth: int) -> None:
        self.deallocate()
        for shader in (self._smoother, self._residual_shader, self._restrict_shader, self._prolong_shader):
            shader.allocate()
        size = (width, height, depth)
        while min(-(-axis // 2) for axis in size) >= SMALLEST:
            residual = Texture3D()
            residual.allocate(*size, GL_R32F)
            self._residuals.append(residual)
            size = tuple(-(-axis // 2) for axis in size)
            correction, source = Texture3D(), Texture3D()
            correction.allocate(*size, GL_R32F)
            source.allocate(*size, GL_R32F)
            self._corrections.append(correction)
            self._sources.append(source)

    def deallocate(self) -> None:
        for volume in self._residuals + self._corrections + self._sources:
            volume.deallocate()
        self._residuals, self._corrections, self._sources = [], [], []
        for shader in (self._smoother, self._residual_shader, self._restrict_shader, self._prolong_shader):
            shader.deallocate()

    def reload(self) -> None:
        for shader in (self._smoother, self._residual_shader, self._restrict_shader, self._prolong_shader):
            shader.reload()

    @property
    def levels(self) -> int:
        """The levels below the finest."""
        return len(self._corrections)

    def solve(self, pressure: Texture3D, divergence: Texture3D, obstacle: Texture3D, cell: float, cycles: int,
              has_obstacles: bool, closed: tuple[bool, bool, bool]) -> None:
        """`cycles` V-cycles on `pressure` in place, `divergence` its right-hand side, `cell` the finest cell."""
        for _ in range(cycles):
            self._cycle(0, pressure, divergence, obstacle, cell, has_obstacles, closed)

    def _cycle(self, level: int, x: Texture3D, b: Texture3D, obstacle: Texture3D, cell: float, has_obstacles: bool,
               closed: tuple[bool, bool, bool]) -> None:
        if level == len(self._corrections):
            self._smoother.solve(x, b, obstacle, cell, COARSEST_SWEEPS, 1.0, False, closed)
            glMemoryBarrier(_BARRIER)
            return
        self._smoother.solve(x, b, obstacle, cell, SMOOTHING, 1.0, has_obstacles, closed)
        glMemoryBarrier(_BARRIER)
        residual = self._residuals[level]
        self._residual_shader.use(x, b, obstacle, residual, cell, has_obstacles, closed)
        glMemoryBarrier(_BARRIER)
        correction, source = self._corrections[level], self._sources[level]
        self._restrict_shader.use(residual, source)
        correction.clear()
        glMemoryBarrier(_BARRIER)
        self._cycle(level + 1, correction, source, obstacle, cell * 2.0, False, closed)
        self._prolong_shader.use(correction, x, closed)
        glMemoryBarrier(_BARRIER)
        self._smoother.solve(x, b, obstacle, cell, SMOOTHING, 1.0, has_obstacles, closed)
        glMemoryBarrier(_BARRIER)
