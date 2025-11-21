from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

import numpy as np

from modules.pose.features import PoseFeature as PoseFeatureUnion

class PoseFeature(Shader):
    def __init__(self) -> None:
        super().__init__()
        self.shader_name = self.__class__.__name__

    def allocate(self, monitor_file = False) -> None:
        super().allocate(self.shader_name, monitor_file)

    def use(self, fbo: int, feature: PoseFeatureUnion, range_scale: float = 1.0, line_thickness: float = 0.1, line_smooth: float = 0.01,
            color=(0.0, 0.5, 1.0, 1.0), bg_color_odd=(1.0, 0.2, 0.0, 1.0), bg_color_even=(1.0, 0.2, 0.0, 1.0)) -> None:
        super().use()
        if not self.allocated: return
        if not fbo: return

        # Flatten values and scores to pass to shader
        values: np.ndarray = np.nan_to_num(feature.values.astype(np.float32), nan=0.0)
        scores: np.ndarray = feature.scores.astype(np.float32)
        min_range: float = feature.default_range()[0] * range_scale
        max_range: float = feature.default_range()[1] * range_scale

        # Create buffer objects
        vbo_values = glGenBuffers(1)
        vbo_scores = glGenBuffers(1)

        # Create texture buffers
        tex_values = glGenTextures(1)
        tex_scores = glGenTextures(1)

        # # Setup values buffer
        glBindBuffer(GL_TEXTURE_BUFFER, vbo_values)
        glBufferData(GL_TEXTURE_BUFFER, values.nbytes, values, GL_STATIC_DRAW)
        glBindTexture(GL_TEXTURE_BUFFER, tex_values)
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, vbo_values)

        # # Setup scores buffer
        glBindBuffer(GL_TEXTURE_BUFFER, vbo_scores)
        glBufferData(GL_TEXTURE_BUFFER, scores.nbytes, scores, GL_STATIC_DRAW)
        glBindTexture(GL_TEXTURE_BUFFER, tex_scores)
        glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, vbo_scores)

        s = self.shader_program
        glUseProgram(s)

        # Pass uniforms to shader
        glUniform1i(glGetUniformLocation(s, "num_joints"), len(feature))
        glUniform1f(glGetUniformLocation(s, "value_min"), min_range)
        glUniform1f(glGetUniformLocation(s, "value_max"), max_range)
        glUniform1f(glGetUniformLocation(s, "line_thickness"), line_thickness)
        glUniform1f(glGetUniformLocation(s, "line_smooth"), line_smooth)

        # Pass color uniforms
        glUniform4f(glGetUniformLocation(s, "color"), *color)
        glUniform4f(glGetUniformLocation(s, "bg_color_odd"), *bg_color_odd)
        glUniform4f(glGetUniformLocation(s, "bg_color_even"), *bg_color_even)

        # Bind texture units
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_BUFFER, tex_values)
        glUniform1i(glGetUniformLocation(s, "values_buffer"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_BUFFER, tex_scores)
        glUniform1i(glGetUniformLocation(s, "scores_buffer"), 1)

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
        glDeleteTextures(2, [tex_values, tex_scores])
        glDeleteBuffers(2, [vbo_values, vbo_scores])

