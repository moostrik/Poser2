from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

import numpy as np

from modules.pose.features.PoseFeatureBase import PoseFeatureBase

class PoseFeature(Shader):
    def __init__(self) -> None:
        super().__init__()
        self.shader_name = self.__class__.__name__

    def allocate(self, monitor_file = False) -> None:
        super().allocate(self.shader_name, monitor_file)

    def use(self, fbo: int, feature: PoseFeatureBase, value_range: tuple[float, float]) -> None:
        super().use()
        if not self.allocated: return
        if not fbo: return

        # Flatten values and scores to pass to shader
        values: np.ndarray = np.nan_to_num(feature.values.astype(np.float32), nan=0.0)
        scores: np.ndarray = feature.scores.astype(np.float32)

        # Create VBO for values
        vbo_values = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_values)
        glBufferData(GL_ARRAY_BUFFER, values.nbytes, values, GL_STATIC_DRAW)

        # Create VBO for scores
        vbo_scores = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_scores)
        glBufferData(GL_ARRAY_BUFFER, scores.nbytes, scores, GL_STATIC_DRAW)

        s = self.shader_program
        glUseProgram(s)

        # Pass uniforms to shader
        glUniform1i(glGetUniformLocation(s, "num_joints"), len(feature))
        glUniform1f(glGetUniformLocation(s, "value_min"), value_range[0])
        glUniform1f(glGetUniformLocation(s, "value_max"), value_range[1])

        # Bind texture units for values and scores
        glActiveTexture(GL_TEXTURE0)
        glBindBuffer(GL_TEXTURE_BUFFER, vbo_values)
        glUniform1i(glGetUniformLocation(s, "values_buffer"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindBuffer(GL_TEXTURE_BUFFER, vbo_scores)
        glUniform1i(glGetUniformLocation(s, "scores_buffer"), 1)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

        # Cleanup
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDeleteBuffers(2, [vbo_values, vbo_scores])

