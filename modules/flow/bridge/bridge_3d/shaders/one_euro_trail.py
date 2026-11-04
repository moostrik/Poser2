"""One Euro Trail shader: one step of the velocity trail as a 1€ filter on x/y."""

from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE1, GL_TEXTURE2, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUniform1i, glUseProgram
from modules.gl import Shader, draw_quad, Texture


class OneEuroTrail(Shader):
    """Writes the trail's next step (RGB): x/y with the cutoff min_cutoff + beta * rate, z by `release`."""

    def use(self, previous: Texture, velocity: Texture, rate: Texture, min_cutoff: float, beta: float, dt: float, release: float) -> None:
        """Advance the trail.

        Args:
            previous: The trail so far, xyz in RGB
            velocity: The new velocity, xyz in RGB
            rate: The x/y velocity's rate of change in R, widths/s²
            min_cutoff: Cutoff at rest, Hz
            beta: Cutoff added per widths/s² of rate, Hz
            dt: Seconds this step
            release: What remains of z
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
        glUniform1f(self.get_uniform_loc("min_cutoff"), min_cutoff)
        glUniform1f(self.get_uniform_loc("beta"), beta)
        glUniform1f(self.get_uniform_loc("dt"), dt)
        glUniform1f(self.get_uniform_loc("release"), release)

        draw_quad()
