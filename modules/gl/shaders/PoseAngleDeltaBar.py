from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

import numpy as np

from modules.pose.features import Angles, AngleVelocity

class PoseAngleDeltaBar(Shader):
    def __init__(self) -> None:
        super().__init__()
        self.shader_name = self.__class__.__name__

    def allocate(self, monitor_file = False) -> None:
        super().allocate(self.shader_name, monitor_file)

    def use(self, fbo: int, angles: Angles, deltas: AngleVelocity, line_thickness: float = 0.1, line_smooth: float = 0.01,
            color_odd=(1.0, 0.2, 0.0, 1.0), color_even=(1.0, 0.2, 0.0, 1.0)) -> None:
        super().use()
        if not self.allocated: return
        if not fbo: return

        # Flatten values and scores to pass to shader
        angle_values: np.ndarray = np.nan_to_num(angles.values.astype(np.float32), nan=0.0)
        delta_values: np.ndarray = np.nan_to_num(deltas.values.astype(np.float32), nan=0.0)
        scores: np.ndarray = np.minimum(angles.scores, deltas.scores).astype(np.float32)
        combined = np.stack([angle_values, delta_values, scores], axis=1).astype(np.float32)

        # Create a single buffer and texture for combined data
        vbo = glGenBuffers(1)
        tex = glGenTextures(1)

        glBindBuffer(GL_TEXTURE_BUFFER, vbo)
        glBufferData(GL_TEXTURE_BUFFER, combined.nbytes, combined, GL_STATIC_DRAW)
        glBindTexture(GL_TEXTURE_BUFFER, tex)
        glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, vbo)

        s = self.shader_program
        glUseProgram(s)

        # Pass uniforms to shader
        glUniform1i(glGetUniformLocation(s, "num_joints"), len(angles))
        glUniform1f(glGetUniformLocation(s, "value_min"), angles.default_range()[0])
        glUniform1f(glGetUniformLocation(s, "value_max"), angles.default_range()[1])
        glUniform1f(glGetUniformLocation(s, "line_thickness"), line_thickness)
        glUniform1f(glGetUniformLocation(s, "line_smooth"), line_smooth)
        glUniform4f(glGetUniformLocation(s, "color_odd"), *color_odd)
        glUniform4f(glGetUniformLocation(s, "color_even"), *color_even)

        # Bind the combined texture buffer to texture unit 0 and set uniform
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_BUFFER, tex)
        glUniform1i(glGetUniformLocation(s, "combined_buffer"), 0)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

        # Cleanup: unbind and delete texture/buffer
        glBindTexture(GL_TEXTURE_BUFFER, 0)
        glBindBuffer(GL_TEXTURE_BUFFER, 0)
        glDeleteTextures(1, [tex])
        glDeleteBuffers(1, [vbo])

