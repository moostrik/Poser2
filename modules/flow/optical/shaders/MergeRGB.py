"""RGB to Luminance shader.

Converts RGB image to single-channel luminance for optical flow input.
Ported from ofxFlowTools ftRGB2LuminanceShader.h
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class MergeRGB(Shader):
    """Convert RGB texture to luminance (grayscale) with optional Y-flip for Image textures."""

    def use(self, source_tex: Texture) -> None:
        """Render luminance to FBO.

        Args:
            target_fbo: Target framebuffer
            source_tex: Source RGB texture
        """
        if not self.allocated or not self.shader_program:
            print("Luminance shader not allocated or shader program missing.")
            return
        if not source_tex.allocated:
            print("Luminance shader: input texture not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, source_tex.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("tex0"), 0)

        # Render
        draw_quad()
