"""CausticNet - the bright net sunlight makes under a wavy surface, a tileable 2D texture, its mean 1."""
from __future__ import annotations

from OpenGL.GL import GL_R16F, glUniform1f, glUniform2i, glUseProgram

from modules.gl import Texture, ComputeShader


class CausticNet(ComputeShader):
    """Draws the net at a moment into a texture; the composite and the rays read it tiled."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16
    WORKGROUP_SIZE_Z = 1

    def use(self, net: Texture, time: float) -> None:
        """The net at `time` (seconds × its speed) into `net` (R16F)."""
        if not self.allocated or not self.shader_program:
            return
        glUseProgram(self.shader_program)
        self.bind_image_write(0, net, GL_R16F)
        glUniform2i(self.get_uniform_loc("uSize"), net.width, net.height)
        glUniform1f(self.get_uniform_loc("uTime"), time)
        self.dispatch(net.width, net.height)
