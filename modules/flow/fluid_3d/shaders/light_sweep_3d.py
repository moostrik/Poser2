"""LightSweep3D - a sun's light through the box in one sweep of parallel lines, one density read a light cell."""
from __future__ import annotations

import math

from OpenGL.GL import GL_RGBA16F, glDispatchCompute, glUniform1f, glUniform1i, glUniform2f, glUniform2i, glUniform3f, glUniform3i, glUseProgram

from modules.gl import Texture3D, ComputeShader


def sweep_lines(size: tuple[int, int, int], toward: tuple[float, float, float]) -> tuple[int, int, tuple[float, float], tuple[int, int], tuple[int, int], float]:
    """The sweep of a sun toward `toward` (a unit vector, x, y, z as the grid's axes) over a grid of `size`
    cubic cells: the sweep axis, the light's direction along it (+1 toward higher cells), the cells slid
    across the other two axes per cell along it, the first line and the number of lines across each (enough
    to surround every cell on the side the light slides in from), and a step's length along the line in cells."""
    travel = tuple(-c for c in toward)
    axis: int = max(range(3), key=lambda a: abs(travel[a]))
    along: float = travel[axis]
    first, second = (a for a in range(3) if a != axis)
    slope: tuple[float, float] = (travel[first] / abs(along), travel[second] / abs(along))
    n: int = size[axis]
    starts: list[int] = []
    counts: list[int] = []
    for s, a in zip(slope, (first, second)):
        start: int = -int(math.ceil(max(s, 0.0) * n)) - 1
        end: int = size[a] + int(math.ceil(max(-s, 0.0) * n)) + 1
        starts.append(start)
        counts.append(end - start)
    return axis, (1 if along > 0 else -1), slope, (starts[0], starts[1]), (counts[0], counts[1]), 1.0 / abs(along)


def sweep_size(size: tuple[int, int, int], toward: tuple[float, float, float]) -> tuple[int, int, int]:
    """The sheared grid a sun's sweep over a grid of `size` writes: a line a column, its layers along the sweep axis."""
    axis, _, _, _, count, _ = sweep_lines(size, toward)
    first, second = (a for a in range(3) if a != axis)
    dims: list[int] = [0, 0, 0]
    dims[axis], dims[first], dims[second] = size[axis], count[0], count[1]
    return (dims[0], dims[1], dims[2])


def sweep_map(size: tuple[int, int, int], toward: tuple[float, float, float]) -> tuple[float, ...]:
    """The affine map, row-major 4 × 4, from a place's texture coordinates in the unsheared grid of `size` to its
    texture coordinates in the sheared grid the sweep writes: along the sweep axis the same, across the line
    through it, line i passing i + 0.5 + slope × k cells across at layer k, as a column of the sheared grid."""
    axis, direction, slope, start, count, _ = sweep_lines(size, toward)
    n: float = float(size[axis])
    along_scale, along_offset = (n, -0.5) if direction > 0 else (-n, n - 0.5)     # the layer k at uvw[axis]
    rows: list[list[float]] = [[0.0, 0.0, 0.0, 0.0] for _ in range(4)]
    rows[3][3] = 1.0
    rows[axis][axis] = 1.0
    others = [a for a in range(3) if a != axis]
    for b, s, first, lines in zip(others, slope, start, count):
        rows[b][b] = size[b] / lines
        rows[b][axis] = -s * along_scale / lines
        rows[b][3] = (-s * along_offset - first) / lines
    return tuple(v for row in rows for v in row)


class LightSweep3D(ComputeShader):
    """A sun's transmittance at every light cell, the density's amount on the way and the water's fog, written
    sheared: a line a column (`sweep_size`, read through `sweep_map`)."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 1

    def use(self, density: Texture3D, light: Texture3D, size: tuple[int, int, int], *, toward: tuple[float, float, float],
            sigma: tuple[float, float, float], fog: tuple[float, float, float]) -> None:
        """Sweep the sun's light through a light grid of `size` cubic cells.

        Args:
            density: The density (RGBA16F, the amount as alpha)
            light: The sheared light grid (RGBA16F, `sweep_size`), written
            size: The light grid's cells, unsheared
            toward: Toward the sun, a unit vector (x right, y down the screen, z away from the viewer)
            sigma: The density's extinction per width per unit of amount, per channel
            fog: The water's extinction per width, per channel
        """
        if not self.allocated or not self.shader_program:
            return
        axis, direction, slope, start, count, step = sweep_lines(size, toward)
        cell: float = 1.0 / max(size[0], 1)        # a cell's side in widths
        glUseProgram(self.shader_program)
        self.bind_texture_3d(0, density, "uDensity")
        self.bind_image_3d_write(0, light, GL_RGBA16F)
        glUniform3i(self.get_uniform_loc("uSize"), *size)
        glUniform1i(self.get_uniform_loc("uAxis"), axis)
        glUniform1i(self.get_uniform_loc("uDirection"), direction)
        glUniform2f(self.get_uniform_loc("uSlope"), *slope)
        glUniform2i(self.get_uniform_loc("uLineStart"), *start)
        glUniform2i(self.get_uniform_loc("uLines"), *count)
        glUniform3f(self.get_uniform_loc("uSigma"), *(s * cell for s in sigma))
        glUniform3f(self.get_uniform_loc("uFog"), *(f * cell for f in fog))
        glUniform1f(self.get_uniform_loc("uStepCells"), step)
        glDispatchCompute((count[0] + 7) // 8, (count[1] + 7) // 8, 1)
