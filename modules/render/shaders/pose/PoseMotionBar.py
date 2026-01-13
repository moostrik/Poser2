from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
import numpy as np

from modules.pose.features import PoseFeatureType as PoseFeatureUnion

class PoseMotionBar(Shader):
    def use(self, feature: PoseFeatureUnion, line_thickness: float = 0.1, line_smooth: float = 0.01,
            color=(0.0, 0.5, 1.0, 1.0), color_odd=(1.0, 0.2, 0.0, 1.0), color_even=(1.0, 0.2, 0.0, 1.0)) -> None:
        if not self.allocated or not self.shader_program:
            print("PoseMotionBar shader not allocated or shader program missing.")
            return

        # Flatten values and scores to pass to shader
        values: np.ndarray = np.nan_to_num(feature.values.astype(np.float32), nan=0.0)
        scores: np.ndarray = feature.scores.astype(np.float32)
        min_range: float = feature.range()[0]
        max_range: float = feature.range()[1]
        min_range = max(min_range, -10.0)
        max_range = min(max_range, 10.0)

        # Activate shader program
        glUseProgram(self.shader_program)

        # Configure shader uniforms
        glUniform1fv(glGetUniformLocation(self.shader_program, "values"), len(values), values)
        glUniform1fv(glGetUniformLocation(self.shader_program, "scores"), len(scores), scores)
        glUniform1i(glGetUniformLocation(self.shader_program, "num_joints"), len(feature))
        glUniform1f(glGetUniformLocation(self.shader_program, "value_min"), min_range)
        glUniform1f(glGetUniformLocation(self.shader_program, "value_max"), max_range)
        glUniform1f(glGetUniformLocation(self.shader_program, "line_thickness"), line_thickness)
        glUniform1f(glGetUniformLocation(self.shader_program, "line_smooth"), line_smooth)
        glUniform4f(glGetUniformLocation(self.shader_program, "color"), *color)
        glUniform4f(glGetUniformLocation(self.shader_program, "color_odd"), *color_odd)
        glUniform4f(glGetUniformLocation(self.shader_program, "color_even"), *color_even)

        # Render
        draw_quad()

        # Cleanup
        glUseProgram(0)

