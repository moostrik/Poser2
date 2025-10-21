from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

class HDT_LineFields(Shader):
    def __init__(self) -> None:
        super().__init__()
        self.shader_name = self.__class__.__name__

    def allocate(self, monitor_file = False) -> None:
        super().allocate(self.shader_name, monitor_file)

    def use(self, fbo,
            cam_tex_0, cam_tex_1, cam_tex_2, line_tex_l, line_tex_r,
            visibility_0: float, visibility_1: float, visibility_2: float,
            cam_color_0: tuple[float, float, float, float], cam_color_1: tuple[float, float, float, float], cam_color_2: tuple[float, float, float, float],
            param_0: float, param_1: float, param_2) -> None :
        super().use()
        if not self.allocated: return
        if not fbo: return

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, cam_tex_0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, cam_tex_1)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, cam_tex_2)
        glActiveTexture(GL_TEXTURE3)
        glBindTexture(GL_TEXTURE_2D, line_tex_l)
        glActiveTexture(GL_TEXTURE4)
        glBindTexture(GL_TEXTURE_2D, line_tex_r)

        s = self.shader_program
        glUseProgram(s)
        glUniform1i(glGetUniformLocation(s, "cam_tex_0"), 0)
        glUniform1i(glGetUniformLocation(s, "cam_tex_1"), 1)
        glUniform1i(glGetUniformLocation(s, "cam_tex_2"), 2)
        glUniform1i(glGetUniformLocation(s, "line_tex_l"), 3)
        glUniform1i(glGetUniformLocation(s, "line_tex_r"), 4)
        glUniform4f(glGetUniformLocation(s, "cam_color_0"), *cam_color_0)
        glUniform4f(glGetUniformLocation(s, "cam_color_1"), *cam_color_1)
        glUniform4f(glGetUniformLocation(s, "cam_color_2"), *cam_color_2)
        glUniform1f(glGetUniformLocation(s, "visibility_0"), visibility_0)
        glUniform1f(glGetUniformLocation(s, "visibility_1"), visibility_1)
        glUniform1f(glGetUniformLocation(s, "visibility_2"), visibility_2)
        glUniform1f(glGetUniformLocation(s, "param0"), param_0)
        glUniform1f(glGetUniformLocation(s, "param1"), param_1)
        glUniform1f(glGetUniformLocation(s, "param2"), param_2)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

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
