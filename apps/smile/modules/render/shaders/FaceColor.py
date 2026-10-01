"""The camera in black and white, with the detected faces' boxes darker and colour inside one of them
(the chosen face)."""

from typing import Sequence

from OpenGL.GL import *  # type: ignore
from modules.gl import Shader, Texture, draw_quad

import logging
logger = logging.getLogger(__name__)

MAX_FIELDS: int = 16    # as in the shader; more faces than this: the first ones

Box = tuple[float, float, float, float]


class FaceColor(Shader):
    def use(self, tex0: Texture, x: float, y: float, w: float, h: float, saturation: float, rise: bool = False,
            fields: Sequence[Box] = (), dark: float = 1.0) -> None:
        """Draw tex0 in black and white, with `saturation` (0..1) of its colour inside the rectangle.
        With `rise` the colour is full and fills that part of the rectangle, from its bottom up.
        `fields` (boxes as the rectangle) and the rectangle where it has no colour are drawn at
        brightness `dark`.

        x, y, w, h: normalised 0..1, top left origin (as DrawRectangleOutline); w or h 0 for no rectangle.
        """
        if not self.allocated or not self.shader_program:
            logger.warning("FaceColor shader not allocated or shader program missing.")
            return
        if not tex0.allocated:
            return

        fields = fields[:MAX_FIELDS]
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform4f(self.get_uniform_loc("rect"), x, y, w, h)
        glUniform1f(self.get_uniform_loc("saturation"), saturation)
        glUniform1f(self.get_uniform_loc("rise"), 1.0 if rise else 0.0)
        if fields:
            glUniform4fv(self.get_uniform_loc("fields"), len(fields), [value for box in fields for value in box])
        glUniform1i(self.get_uniform_loc("field_count"), len(fields))
        glUniform1f(self.get_uniform_loc("dark"), dark)
        draw_quad()
