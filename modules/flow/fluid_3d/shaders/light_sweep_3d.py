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


class LightSweep3D(ComputeShader):
    """A sun's transmittance at every light cell: the density's amount on the way and the water's fog."""

    WORKGROUP_SIZE_X = 8
    WORKGROUP_SIZE_Y = 8
    WORKGROUP_SIZE_Z = 1

    def use(self, density: Texture3D, light: Texture3D, *, toward: tuple[float, float, float],
            sigma: tuple[float, float, float], fog: tuple[float, float, float]) -> None:
        """Sweep the sun's light through the light grid.

        Args:
            density: The density (RGBA16F, the amount as alpha)
            light: The light grid (RGBA16F), written; its cells cubic
            toward: Toward the sun, a unit vector (x right, y down the screen, z away from the viewer)
            sigma: The density's extinction per width per unit of amount, per channel
            fog: The water's extinction per width, per channel
        """
        if not self.allocated or not self.shader_program:
            return
        size: tuple[int, int, int] = (light.width, light.height, light.depth)
        axis, direction, slope, start, count, step = sweep_lines(size, toward)
        cell: float = 1.0 / max(light.width, 1)        # a cell's side in widths
        glUseProgram(self.shader_program)
        self.bind_texture_3d(0, density, "uDensity")
        self.bind_image_3d_write(0, light, GL_RGBA16F)
        glUniform3i(self.get_uniform_loc("uSize"), *size)
        glUniform1i(self.get_uniform_loc("uAxis"), axis)
        glUniform1i(self.get_uniform_loc("uDirection"), direction)
        glUniform2f(self.get_uniform_loc("uSlope"), *slope)
        glUniform2i(self.get_uniform_loc("uLineStart"), *start)
        glUniform3f(self.get_uniform_loc("uSigma"), *(s * cell for s in sigma))
        glUniform3f(self.get_uniform_loc("uFog"), *(f * cell for f in fog))
        glUniform1f(self.get_uniform_loc("uStepCells"), step)
        glDispatchCompute((count[0] + 6) // 7, (count[1] + 6) // 7, 1)     # 8 × 8 lines a workgroup, overlapping by one
