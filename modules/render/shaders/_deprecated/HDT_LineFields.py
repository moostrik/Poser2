from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Fbo, Texture

class HDT_LineFields(Shader):
    def use(self, fbo: Fbo,
            cam_tex_0: Texture, cam_tex_1: Texture, cam_tex_2: Texture, line_tex_l: Texture, line_tex_r: Texture,
            visibility_0: float, visibility_1: float, visibility_2: float,
            cam_color_0: tuple[float, float, float, float], cam_color_1: tuple[float, float, float, float], cam_color_2: tuple[float, float, float, float],
            param_0: float, param_1: float, param_2: float) -> None:
        if not self.allocated or not self.shader_program: return
        if not fbo.allocated: return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Set up render target
        glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo_id)
        glViewport(0, 0, fbo.width, fbo.height)

        # Bind input textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, cam_tex_0.tex_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, cam_tex_1.tex_id)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, cam_tex_2.tex_id)
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, line_tex_l.tex_id)
        glActiveTexture(GL_TEXTURE4)
        glBindTexture(GL_TEXTURE_2D, line_tex_r.tex_id)

        # Configure shader uniforms
        glUniform2f(glGetUniformLocation(self.shader_program, "resolution"), float(fbo.width), float(fbo.height))
        glUniform1i(glGetUniformLocation(self.shader_program, "cam_tex_0"), 0)
        glUniform1i(glGetUniformLocation(self.shader_program, "cam_tex_1"), 1)
        glUniform1i(glGetUniformLocation(self.shader_program, "cam_tex_2"), 2)
        glUniform1i(glGetUniformLocation(self.shader_program, "line_tex_l"), 3)
        glUniform1i(glGetUniformLocation(self.shader_program, "line_tex_r"), 4)
        glUniform4f(glGetUniformLocation(self.shader_program, "cam_color_0"), *cam_color_0)
        glUniform4f(glGetUniformLocation(self.shader_program, "cam_color_1"), *cam_color_1)
        glUniform4f(glGetUniformLocation(self.shader_program, "cam_color_2"), *cam_color_2)
        glUniform1f(glGetUniformLocation(self.shader_program, "visibility_0"), visibility_0)
        glUniform1f(glGetUniformLocation(self.shader_program, "visibility_1"), visibility_1)
        glUniform1f(glGetUniformLocation(self.shader_program, "visibility_2"), visibility_2)
        glUniform1f(glGetUniformLocation(self.shader_program, "param0"), param_0)
        glUniform1f(glGetUniformLocation(self.shader_program, "param1"), param_1)
        glUniform1f(glGetUniformLocation(self.shader_program, "param2"), param_2)

        # Render
        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE4)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)
