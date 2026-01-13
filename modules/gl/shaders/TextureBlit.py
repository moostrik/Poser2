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

    def use(self, texture: Texture, x: float, y: float, w: float, h: float) -> None:
        """Draw texture to screen region.

        Args:
            texture: Texture to draw
            x: X position in pixels (top-left origin)
            y: Y position in pixels (top-left origin)
            w: Width in pixels
            h: Height in pixels
        """
        if not self.allocated or not self.shader_program: return
        if not texture.allocated: return

        # Get current viewport to determine screen dimensions
        viewport = glGetIntegerv(GL_VIEWPORT)
        screen_width, screen_height = viewport[2], viewport[3]

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind to screen framebuffer (already bound, but explicit for clarity)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture.tex_id)

        # Configure shader uniforms - use screen resolution, not region size
        glUniform2f(glGetUniformLocation(self.shader_program, "resolution"),
                    float(screen_width), float(screen_height))
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)

        # Render quad at specified position in pixel space
        draw_quad_pixels_at(x, y, w, h)
