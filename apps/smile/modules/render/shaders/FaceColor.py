"""The camera in black and white, with colour inside one rectangle (the chosen face)."""

from OpenGL.GL import *  # type: ignore
from modules.gl import Shader, Texture, draw_quad

import logging
logger = logging.getLogger(__name__)


class FaceColor(Shader):
    def use(self, tex0: Texture, x: float, y: float, w: float, h: float, saturation: float) -> None:
        """Draw tex0 in black and white, with `saturation` (0..1) of its colour inside the rectangle.

        x, y, w, h: normalised 0..1, top left origin (as DrawRectangleOutline); w or h 0 for no rectangle.
        """
        if not self.allocated or not self.shader_program:
            logger.warning("FaceColor shader not allocated or shader program missing.")
            return
        if not tex0.allocated:
            return

        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform4f(self.get_uniform_loc("rect"), x, y, w, h)
        glUniform1f(self.get_uniform_loc("saturation"), saturation)
        draw_quad()
