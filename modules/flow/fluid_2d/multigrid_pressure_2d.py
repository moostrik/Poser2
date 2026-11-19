"""The 2D pressure by a geometric multigrid, as the 3D's (`fluid_3d/multigrid_pressure.py`) on the plane: V-cycles
over the grid halved down to a few cells, red-black Gauss–Seidel smoothing each level, warm from the last step's
pressure (McAdams, Sifakis, Teran 2010). A level halves only the axes whose cells are shorter than the longest by more
than COARSEN (semi-coarsening), so the cells grow toward square before the other axis is halved."""

from OpenGL.GL import GL_R32F, GL_SHADER_IMAGE_ACCESS_BARRIER_BIT, GL_TEXTURE_FETCH_BARRIER_BIT, glMemoryBarrier

from modules.gl import Fbo, Texture
from .shaders import SorPressure2D, ResidualRestrict2D, CoarsePressure2D, Prolong2D

_BARRIER: int = int(GL_TEXTURE_FETCH_BARRIER_BIT) | int(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

SMOOTHING: int = 2          # red-black Gauss–Seidel sweeps before and after the coarser level, on each level
COARSEST_SWEEPS: int = 16   # sweeps on the coarsest level, which stands for its solve
SMALLEST: int = 4           # an axis is halved while it stays at least this many cells
COARSEN: float = 1.5        # an axis is halved while its cell is this much shorter than the longest, or both are near square


def coarsening(size: tuple[int, int], cell: tuple[float, float]) -> tuple[int, int]:
    """Per axis, 2 where the next coarser level halves it, 1 where it keeps it: the axes whose cells are shorter
    than the longest by more than COARSEN, both once the cells are near square; an axis is kept once halving would
    leave it under SMALLEST. (1, 1): no coarser level."""
    longest: float = max(cell)
    near_square: bool = all(side * COARSEN >= longest for side in cell)
    wanted = [near_square or side * COARSEN < longest for side in cell]
    halves = [2 if want and -(-n // 2) >= SMALLEST else 1 for want, n in zip(wanted, size)]
    if halves == [1, 1] and not near_square:        # the short axis is spent: halve what still can be
        halves = [2 if -(-n // 2) >= SMALLEST else 1 for n in size]
    return (halves[0], halves[1])


class MultigridPressure2D:
    """V-cycles on the 2D pressure in place: on each level smooth, the residual onto the next coarser level as its
    right-hand side, the coarser level's correction solved from 0 and added back, smooth again. The walls are the
    solvers' on every level; obstacles are kept on the finest level's smoothing only."""

    def __init__(self) -> None:
        self._smoother: SorPressure2D = SorPressure2D()
        self._residual_restrict_shader: ResidualRestrict2D = ResidualRestrict2D()     # the residual, restricted in one pass
        self._coarse_shader: CoarsePressure2D = CoarsePressure2D()     # the coarsest level's sweeps in one dispatch
        self._prolong_shader: Prolong2D = Prolong2D()
        self._corrections: list[Fbo] = []           # each coarser level's correction
        self._sources: list[Fbo] = []               # each coarser level's right-hand side, the finer residual restricted
        self._factors: list[tuple[int, int]] = []   # each coarser level's halving of the level above, per axis

    def allocate(self, width: int, height: int, cell: tuple[float, float]) -> None:
        """The levels for a grid of `width` × `height` cells of `cell`'s sides (widths)."""
        self.deallocate()
        for shader in self._shaders:
            shader.allocate()
        size = (width, height)
        while True:
            factor = coarsening(size, cell)
            if factor == (1, 1):
                break
            size = (-(-size[0] // factor[0]), -(-size[1] // factor[1]))
            cell = (cell[0] * factor[0], cell[1] * factor[1])
            correction, source = Fbo(), Fbo()
            correction.allocate(*size, GL_R32F)
            source.allocate(*size, GL_R32F)
            self._corrections.append(correction)
            self._sources.append(source)
            self._factors.append(factor)

    def deallocate(self) -> None:
        for level in self._corrections + self._sources:
            level.deallocate()
        self._corrections, self._sources, self._factors = [], [], []
        for shader in self._shaders:
            shader.deallocate()

    def reload(self) -> None:
        for shader in self._shaders:
            shader.reload()

    @property
    def _shaders(self) -> tuple:
        return (self._smoother, self._residual_restrict_shader, self._coarse_shader, self._prolong_shader)

    @property
    def levels(self) -> int:
        """The levels below the finest."""
        return len(self._corrections)

    def sor(self, pressure: Texture, divergence: Texture, obstacle: Texture, cell: tuple[float, float], sweeps: int,
            omega: float, has_obstacles: bool, closed: bool) -> None:
        """`sweeps` red-black SOR sweeps on `pressure` in place at `omega`, the solver `SOR` on its own."""
        self._smoother.solve(pressure, divergence, obstacle, cell, sweeps, omega, has_obstacles, closed)
        glMemoryBarrier(_BARRIER)

    def solve(self, pressure: Texture, divergence: Texture, obstacle: Texture, cell: tuple[float, float], cycles: int,
              has_obstacles: bool, closed: bool) -> None:
        """`cycles` V-cycles on `pressure` in place, `divergence` its right-hand side, `cell` the finest cell."""
        for _ in range(cycles):
            self._cycle(0, pressure, divergence, obstacle, cell, has_obstacles, closed)

    def _cycle(self, level: int, x: Texture, b: Texture, obstacle: Texture, cell: tuple[float, float], has_obstacles: bool,
               closed: bool) -> None:
        if level == len(self._corrections):
            if self._coarse_shader.fits(x):
                self._coarse_shader.solve(x, b, cell, COARSEST_SWEEPS, closed)
            else:
                self._smoother.solve(x, b, obstacle, cell, COARSEST_SWEEPS, 1.0, False, closed)
            glMemoryBarrier(_BARRIER)
            return
        self._smoother.solve(x, b, obstacle, cell, SMOOTHING, 1.0, has_obstacles, closed)
        glMemoryBarrier(_BARRIER)
        correction, source = self._corrections[level], self._sources[level]
        factor = self._factors[level]
        self._residual_restrict_shader.use(x, b, obstacle, source, cell, factor, has_obstacles, closed)
        correction.clear()
        glMemoryBarrier(_BARRIER)
        coarser = (cell[0] * factor[0], cell[1] * factor[1])
        self._cycle(level + 1, correction, source, obstacle, coarser, False, closed)
        self._prolong_shader.use(correction, x, closed)
        glMemoryBarrier(_BARRIER)
        self._smoother.solve(x, b, obstacle, cell, SMOOTHING, 1.0, has_obstacles, closed)
        glMemoryBarrier(_BARRIER)
