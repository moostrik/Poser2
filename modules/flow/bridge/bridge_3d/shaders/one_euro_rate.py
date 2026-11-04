"""One Euro Rate shader: the 1€ filter's smoothed rate of change of the x/y velocity."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class OneEuroRate(Shader):
    """Writes the rate |velocity - previous| / dt (widths/s², R), low-passed with the previous rate by `weight`."""

    def use(self, previous: Texture, velocity: Texture, rate: Texture, dt: float, weight: float) -> None:
        """Advance the rate.

        Args:
            previous: The trail so far, xyz in RGB
            velocity: The new velocity, xyz in RGB
            rate: The previous rate in R
            dt: Seconds this step
            weight: What remains of the previous rate
        """
        if not self.allocated or not self.shader_program:
            return
        if not previous.allocated or not velocity.allocated or not rate.allocated:
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, previous.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, velocity.tex_id)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, rate.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)
        glUniform1i(self.get_uniform_loc("tex2"), 2)
        glUniform1f(self.get_uniform_loc("dt"), dt)
        glUniform1f(self.get_uniform_loc("weight"), weight)

        draw_quad()
