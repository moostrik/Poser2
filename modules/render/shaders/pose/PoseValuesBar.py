from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

import numpy as np

from modules.pose.features import PoseFeatureType as PoseFeatureUnion

class PoseValuesBar(Shader):
    def allocate(self) -> None:
        super().allocate()
        self.vbo_values = glGenBuffers(1)
        self.vbo_colors = glGenBuffers(1)
        self.tex_values = glGenTextures(1)
        self.tex_colors = glGenTextures(1)

    def deallocate(self):
        super().deallocate()
        glDeleteTextures(2, [self.tex_values, self.tex_colors])
        glDeleteBuffers(2, [self.vbo_values, self.vbo_colors])

    def use(self, fbo: int, norm_values: np.ndarray, colors: np.ndarray, line_thickness: float = 0.1, line_smooth: float = 0.01) -> None:
        if not self.allocated: return
        if not fbo: return

        # Flatten values and colors to pass to shader
        values: np.ndarray = np.nan_to_num(norm_values.astype(np.float32), nan=0.0)

        # # Setup values buffer
        glBindBuffer(GL_TEXTURE_BUFFER, self.vbo_values)
        glBufferData(GL_TEXTURE_BUFFER, values.nbytes, values, GL_STATIC_DRAW)
        glBindTexture(GL_TEXTURE_BUFFER, self.tex_values)
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, self.vbo_values)

        # # Setup colors buffer
        glBindBuffer(GL_TEXTURE_BUFFER, self.vbo_colors)
        glBufferData(GL_TEXTURE_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
        glBindTexture(GL_TEXTURE_BUFFER, self.tex_colors)
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, self.vbo_colors)

        s = self.shader_program
        glUseProgram(s)

        # Pass uniforms to shader
        glUniform1i(glGetUniformLocation(s, "num_values"), len(values))
        glUniform1f(glGetUniformLocation(s, "line_thickness"), line_thickness)
        glUniform1f(glGetUniformLocation(s, "line_smooth"), line_smooth)

        # Bind texture units
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_BUFFER, self.tex_values)
        glUniform1i(glGetUniformLocation(s, "values_buffer"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_BUFFER, self.tex_colors)
        glUniform1i(glGetUniformLocation(s, "colors_buffer"), 1)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

        # Cleanup
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_BUFFER, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_BUFFER, 0)
        glBindBuffer(GL_TEXTURE_BUFFER, 0)
