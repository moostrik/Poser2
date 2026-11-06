"""Velocity Event shader.

Peak-holds one event of the motion, a start or a stop, from the fast and slow trails.
"""

from enum import IntEnum

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3, GL_TEXTURE4, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class Event(IntEnum):
    START = 1       # fast - slow rises along the motion
    STOP = -1       # it falls


class VelocityEvent(Shader):
    """Writes max(event * region, previous * decay) to the red channel."""

    def use(self, fast: Texture, slow: Texture, previous: Texture, event: Event, decay: float,
            region_fast: Texture, region_slow: Texture, region_weight: float, lod: float) -> None:
        """Hold one event.

        Args:
            fast: RGB fast trail
            slow: RGB slow trail
            previous: Previous held event
            event: START or STOP
            decay: What remains of the held event this update
            region_fast: Start: the region's mean moving velocity, fast trail (RGB, a texel per region); stop: the
                region speeds, fast in R and slow in G, with mipmaps
            region_slow: Start: the region's mean moving velocity, slow trail; stop: unread
            region_weight: 0 = every pixel's event counts; 1 = a start as far as its region speeds up, a stop as far as it slows down
            lod: Stop: mip level of a region
        """
        if not self.allocated or not self.shader_program:
            return
        if not fast.allocated or not slow.allocated or not previous.allocated:
            return

        glUseProgram(self.shader_program)

        for unit, texture in enumerate((fast, slow, previous, region_fast, region_slow)):
            glActiveTexture((GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3, GL_TEXTURE4)[unit])
            glBindTexture(GL_TEXTURE_2D, texture.tex_id)
            glUniform1i(self.get_uniform_loc(f"tex{unit}"), unit)
        glUniform1f(self.get_uniform_loc("sign"), float(event))
        glUniform1f(self.get_uniform_loc("decay"), decay)
        glUniform1f(self.get_uniform_loc("region_weight"), region_weight)
        glUniform1f(self.get_uniform_loc("lod"), lod)

        draw_quad()
