from __future__ import annotations
from typing import TYPE_CHECKING

from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad_pixels

if TYPE_CHECKING:
    from modules.gl.Fbo import Fbo
    from modules.gl.Texture import Texture

class FboBlit(Shader):
    """Simple shader for blitting textures to FBOs using pixel coordinates.

    Uses generic vertex/fragment shaders for texture sampling.
    """

    def use(self, dst_fbo: Fbo, src_texture: Texture, flip_v: bool = False) -> None:
        """Blit source texture to destination FBO.

        Args:
            dst_fbo: Destination FBO to blit to
            src_texture: Source texture to blit from
            flip_v: Flip texture V coordinate (for normalizing upload orientation)
        """
        if not self.allocated or not self.shader_program: return
        if not dst_fbo.allocated or not src_texture.allocated: return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind to destination framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, dst_fbo.fbo_id)
        glViewport(0, 0, dst_fbo.width, dst_fbo.height)

        # Bind source texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, src_texture.tex_id)

        # Configure shader uniforms
        glUniform2f(glGetUniformLocation(self.shader_program, "resolution"), float(dst_fbo.width), float(dst_fbo.height))
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)
        # flip_v=True means source is staging (top at V=0), so DON'T flip V in shader
        # flip_v=False means source is normalized (top at V=1), use default flipV=true
        glUniform1i(glGetUniformLocation(self.shader_program, "flipV"), int(not flip_v))

        # Render fullscreen quad in pixel space
        draw_quad_pixels(dst_fbo.width, dst_fbo.height)

        # Cleanup
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)
