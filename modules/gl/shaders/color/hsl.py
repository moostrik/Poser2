from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, glActiveTexture, glBindTexture, glUniform1f, glUseProgram
from ... import Shader, draw_quad, Texture

import logging
logger = logging.getLogger(__name__)

class Hsl(Shader):
    def use(self, src: Texture, hue: float, saturation: float, lightness: float) -> None:
        if not self.allocated or not self.shader_program:
            logger.warning("Hsl shader not allocated or shader program missing.")
            return
        if not src.allocated:
            logger.warning("Hsl shader: input texture not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1f(self.get_uniform_loc("hue"), hue)
        glUniform1f(self.get_uniform_loc("saturation"), saturation)
        glUniform1f(self.get_uniform_loc("lightness"), lightness)

        # Render
        draw_quad()

