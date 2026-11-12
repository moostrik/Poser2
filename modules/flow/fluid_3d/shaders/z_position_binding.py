"""The z position an injection centers its slab on (slab.glsl's `slab_center`), bound beside its input."""
from __future__ import annotations

from OpenGL.GL import glUniform1i

from modules.gl import ComputeShader, Texture


def bind_z_position(shader: ComputeShader, unit: int, z_position: Texture | None) -> None:
    """`z_position` (red the z position times its weight, green the weight) on texture `unit` as `uZPosition`,
    or none: a pixel without a z position, or every pixel without one, puts nothing in."""
    has: bool = z_position is not None and z_position.allocated
    if has:
        assert z_position is not None
        shader.bind_texture(unit, z_position, "uZPosition")
    glUniform1i(shader.get_uniform_loc("uHasZPosition"), int(has))
