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
        if not self.allocated or not self.shader_program: return
        if not target_fbo.allocated or not curr_tex.allocated or not prev_tex.allocated: return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Set up render target
        glBindFramebuffer(GL_FRAMEBUFFER, target_fbo.fbo_id)
        glViewport(0, 0, target_fbo.width, target_fbo.height)

        # Bind input textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, curr_tex.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, prev_tex.tex_id)

        # Configure shader uniforms
        glUniform2f(glGetUniformLocation(self.shader_program, "resolution"), float(target_fbo.width), float(target_fbo.height))
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)
        glUniform1i(glGetUniformLocation(self.shader_program, "tex1"), 1)
        glUniform2f(
            glGetUniformLocation(self.shader_program, "offset"),
            float(offset) / target_fbo.width,
            float(offset) / target_fbo.height
        )
        glUniform1f(glGetUniformLocation(self.shader_program, "threshold"), threshold)
        glUniform2f(glGetUniformLocation(self.shader_program, "force"), strength_x, strength_y)
        glUniform1f(glGetUniformLocation(self.shader_program, "power"), power)

        # Render
        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)
