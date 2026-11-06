"""Velocity Event shader.

One event of the motion, a start, bow, stop or wake, from the fast and slow trails.
"""

from enum import IntEnum

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class MotionEvent(IntEnum):
    """The four events: a rise of fast - slow along the motion that its region shares (START) or not (BOW, a
    limb arriving), a fall that its region shares (STOP) or not (WAKE, a limb leaving)."""
    START = 0
    BOW = 1
    STOP = 2
    WAKE = 3

    @property
    def sign(self) -> float:
        """+1 for a rise, -1 for a fall."""
        return 1.0 if self in (MotionEvent.START, MotionEvent.BOW) else -1.0

    @property
    def edge(self) -> bool:
        """Whether it is the part its region does not share."""
        return self in (MotionEvent.BOW, MotionEvent.WAKE)


class VelocityEvent(Shader):
    """Writes the event times its region's share or the rest to the red channel."""

    def use(self, fast: Texture, slow: Texture, event: MotionEvent, scale: float,
            region_fast: Texture, region_slow: Texture, region_weight: float, lod: float) -> None:
        """Write one event.

        Args:
            fast: RGB fast trail
            slow: RGB slow trail
            event: START, BOW, STOP or WAKE
            scale: 1 / the trails' step peak, so a sudden start or stop of speed v reads v; 0 = no events
            region_fast: A rise: the region's mean moving velocity, fast trail (RGB, a texel per region); a fall:
                the region speeds, fast in R and slow in G, with mipmaps
            region_slow: A rise: the region's mean moving velocity, slow trail; a fall: unread
            region_weight: 0 = all of an event is its region's (start, stop); 1 = as far as its region speeds up or
                slows down, the rest the bow or the wake
            lod: A fall: mip level of a region
        """
        if not self.allocated or not self.shader_program:
            return
        if not fast.allocated or not slow.allocated:
            return

        glUseProgram(self.shader_program)

        for unit, texture in enumerate((fast, slow, region_fast, region_slow)):
            glActiveTexture((GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE3)[unit])
            glBindTexture(GL_TEXTURE_2D, texture.tex_id)
            glUniform1i(self.get_uniform_loc(f"tex{unit}"), unit)
        glUniform1f(self.get_uniform_loc("sign"), event.sign)
        glUniform1f(self.get_uniform_loc("edge"), 1.0 if event.edge else 0.0)
        glUniform1f(self.get_uniform_loc("scale"), scale)
        glUniform1f(self.get_uniform_loc("region_weight"), region_weight)
        glUniform1f(self.get_uniform_loc("lod"), lod)

        draw_quad()
