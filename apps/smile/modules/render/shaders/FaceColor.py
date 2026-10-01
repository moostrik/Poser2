"""The camera in black and white. Every detected face's box shows how far the face is from good
(darker, pixelated, blurred or fogged), and the chosen face's box turns to colour with its charge."""

from typing import Sequence

from OpenGL.GL import *  # type: ignore
from modules.gl import Shader, Texture, draw_quad

import logging
logger = logging.getLogger(__name__)

MAX_FIELDS: int = 16    # as in the shader; more faces than this: the first ones

Field = tuple[float, float, float, float, float]    # x, y, w, h, distance from good (0..1)


class FaceColor(Shader):
    def use(self, tex0: Texture, width: int, height: int, fields: Sequence[Field], effect: int, amount: float,
            rect: Field, charge: float, fill: int, color: tuple[float, float, float, float], blurred: Texture | None = None) -> None:
        """Draw tex0 in black and white into a width x height buffer.

        fields: the faces' boxes with their distance from good, shown by `effect` (index in FaceEffect)
        at strength `amount`; `blurred` is a blurred tex0 for the blur effect.
        rect: the chosen face's box, turned to colour by `charge` (0..1) as `fill` (index in FaceFill)
        says; `color` is the traced outline's. A rect of width or height 0 is no rect.
        Boxes are normalised 0..1, top left origin (as DrawRectangleOutline).
        """
        if not self.allocated or not self.shader_program:
            logger.warning("FaceColor shader not allocated or shader program missing.")
            return
        if not tex0.allocated:
            return

        fields = fields[:MAX_FIELDS]
        glUseProgram(self.shader_program)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, blurred.tex_id if blurred is not None and blurred.allocated else tex0.tex_id)
        glUniform1i(self.get_uniform_loc("tex_blur"), 1)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform2f(self.get_uniform_loc("resolution"), float(width), float(height))

        if fields:
            glUniform4fv(self.get_uniform_loc("fields"), len(fields), [value for field in fields for value in field[:4]])
            glUniform1fv(self.get_uniform_loc("field_far"), len(fields), [field[4] for field in fields])
        glUniform1i(self.get_uniform_loc("field_count"), len(fields))
        glUniform1i(self.get_uniform_loc("effect"), effect)
        glUniform1f(self.get_uniform_loc("amount"), amount)

        glUniform4f(self.get_uniform_loc("rect"), *rect[:4])
        glUniform1f(self.get_uniform_loc("rect_far"), rect[4])
        glUniform1f(self.get_uniform_loc("charge"), charge)
        glUniform1i(self.get_uniform_loc("fill"), fill)
        glUniform4f(self.get_uniform_loc("face_color"), *color)
        draw_quad()
