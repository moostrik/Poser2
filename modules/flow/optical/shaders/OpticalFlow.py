"""Optical Flow shader.

Computes sparse optical flow (Lucas-Kanade style) between two frames.
Ported from ofxFlowTools ftOpticalFlowShader.h
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Fbo, Texture


class OpticalFlow(Shader):
    """Compute optical flow velocity field between two luminance frames."""

    def use(self, target_fbo: Fbo, curr_tex: Texture, prev_tex: Texture,
             offset: int = 3, threshold: float = 0.1, strength_x: float = 3.0, strength_y: float = 3.0,
             power: float = 1.0) -> None:
        """Compute optical flow and render to FBO.

        Args:
            target_fbo: Target framebuffer
            curr_tex: Current frame luminance texture
            prev_tex: Previous frame luminance texture
            offset: Gradient sample offset in pixels (1-10)
            threshold: Motion threshold (0-0.2)
            strength_x: X velocity multiplier (-10.0 to 10.0, negative inverts)
            strength_y: Y velocity multiplier (-10.0 to 10.0, negative inverts)
            power: Power curve for magnitude (related to boost)
        """
        if not self.allocated:
            return
        if self.shader_program is None:
            return

        glBindFramebuffer(GL_FRAMEBUFFER, target_fbo.fbo_id)
        glViewport(0, 0, target_fbo.width, target_fbo.height)
        glDisable(GL_BLEND)

        glUseProgram(self.shader_program)

        # Bind textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, curr_tex.tex_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, prev_tex.tex_id)
        glUniform1i(glGetUniformLocation(self.shader_program, "tex1"), 1)

        # Convert pixel offset to normalized coordinates (shader responsibility)
        glUniform2f(
            glGetUniformLocation(self.shader_program, "offset"),
            float(offset) / target_fbo.width,
            float(offset) / target_fbo.height
        )
        glUniform1f(glGetUniformLocation(self.shader_program, "threshold"), threshold)

        # Strength includes direction (negative = inverted)
        glUniform2f(glGetUniformLocation(self.shader_program, "force"), strength_x, strength_y)

        glUniform1f(glGetUniformLocation(self.shader_program, "power"), power)

        draw_quad()

        glEnable(GL_BLEND)
        glUseProgram(0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
