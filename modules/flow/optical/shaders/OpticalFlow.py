"""Optical Flow shader.

Computes sparse optical flow (Lucas-Kanade style) between two frames.
Ported from ofxFlowTools ftOpticalFlowShader.h
"""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class OpticalFlow(Shader):
    """Compute optical flow velocity field between two luminance frames."""

    def use(self, curr_tex: Texture, prev_tex: Texture,
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
        if not self.allocated or not self.shader_program:
            print("OpticalFlow shader not allocated or shader program missing.")
            return
        if not curr_tex.allocated or not prev_tex.allocated:
            print("OpticalFlow shader: input textures not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, curr_tex.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, prev_tex.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1i(self.get_uniform_loc("tex1"), 1)

        # Convert pixel offset to normalized coordinates
        offset_x = float(offset) / curr_tex.width
        offset_y = float(offset) / curr_tex.height
        glUniform2f(self.get_uniform_loc("offset"), offset_x, offset_y)

        glUniform1f(self.get_uniform_loc("threshold"), threshold)
        glUniform2f(self.get_uniform_loc("force"), strength_x, strength_y)
        glUniform1f(self.get_uniform_loc("power"), power)

        # Render
        draw_quad()
