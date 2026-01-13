from __future__ import annotations
from typing import TYPE_CHECKING

from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad_pixels_at

if TYPE_CHECKING:
    from modules.gl.Texture import Texture

class TextureBlit(Shader):
    """Simple shader for drawing textures to screen using pixel coordinates.

    Uses generic vertex/fragment shaders for texture sampling.
    """

    def use(self, texture: Texture, x: float, y: float, w: float, h: float, flip_v: bool = False) -> None:
        """Draw texture to screen region.

        Args:
            texture: Texture to draw
            x: X position in pixels (top-left origin)
            y: Y position in pixels (top-left origin)
            w: Width in pixels
            h: Height in pixels
            flip_v: Flip texture V coordinate (True for FBO content)
        """
        if not self.allocated or not self.shader_program: return
        if not texture.allocated: return

        # Get current viewport to determine rendering target dimensions
        viewport = glGetIntegerv(GL_VIEWPORT)
        target_width, target_height = viewport[2], viewport[3]

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture.tex_id)

        # Configure shader uniforms - use current rendering target resolution
        glUniform2f(glGetUniformLocation(self.shader_program, "resolution"), float(target_width), float(target_height))
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)
        # flip_v parameter: same inversion as FboBlit (True = don't flip V)
        # Default flip_v=False uses shader default flipV=true (correct for normalized textures)
        glUniform1i(glGetUniformLocation(self.shader_program, "flipV"), int(not flip_v))

        # Render quad at specified position in pixel space
        draw_quad_pixels_at(x, y, w, h)

        # Cleanup - unbind resources
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
