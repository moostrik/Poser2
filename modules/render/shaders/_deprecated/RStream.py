from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad_pixels
from modules.gl import Fbo
import numpy as np

class RStream(Shader):
    def __init__(self) -> None:
        super().__init__()
        self.texture_id = None

    def allocate(self) -> None:
        super().allocate()
        if self.allocated and self.texture_id is None:
            self.texture_id = glGenTextures(1)

    def deallocate(self) -> None:
        super().deallocate()
        if self.texture_id is not None:
            glDeleteTextures(1, [self.texture_id])
            self.texture_id = None

    def use(self, fbo: Fbo, correlation_data: np.ndarray, pair_index: int, total_pairs: int,
            line_color: tuple = (1.0, 1.0, 1.0), line_width: float = 5.0) -> None:
        if not self.allocated or not self.shader_program or not fbo.allocated or self.texture_id is None: return

        # Upload correlation data to texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        # Normalize data to [0, 1] range
        normalized_data = np.clip(correlation_data.astype(np.float32), 0.0, 1.0)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, len(normalized_data), 1, 0,
                     GL_RED, GL_FLOAT, normalized_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # Activate shader program
        glUseProgram(self.shader_program)

        # Set up render target
        glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo_id)
        glViewport(0, 0, fbo.width, fbo.height)

        # Configure shader uniforms
        glUniform2f(glGetUniformLocation(self.shader_program, "resolution"), float(fbo.width), float(fbo.height))
        glUniform1i(glGetUniformLocation(self.shader_program, "correlationTexture"), 0)
        glUniform3f(glGetUniformLocation(self.shader_program, "lineColor"), *line_color)
        glUniform1f(glGetUniformLocation(self.shader_program, "lineWidth"), line_width)
        glUniform2f(glGetUniformLocation(self.shader_program, "viewportSize"), float(fbo.width), float(fbo.height))
        glUniform1i(glGetUniformLocation(self.shader_program, "pairIndex"), pair_index)
        glUniform1i(glGetUniformLocation(self.shader_program, "totalPairs"), total_pairs)

        # Render
        draw_quad_pixels(fbo.width, fbo.height)

        # Cleanup
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)