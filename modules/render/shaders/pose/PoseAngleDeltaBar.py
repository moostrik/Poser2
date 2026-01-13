from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

import numpy as np

from modules.pose.features import Angles, AngleVelocity

class PoseAngleDeltaBar(Shader):
    def use(self, angles: Angles, deltas: AngleVelocity, line_thickness: float = 0.1, line_smooth: float = 0.01,
            color_odd=(1.0, 0.2, 0.0, 1.0), color_even=(1.0, 0.2, 0.0, 1.0)) -> None:
        if not self.allocated or not self.shader_program:
            print("PoseAngleDeltaBar shader not allocated or shader program missing.")
            return

        # Flatten values and scores to pass to shader
        angle_values: np.ndarray = np.nan_to_num(angles.values.astype(np.float32), nan=0.0)
        delta_values: np.ndarray = np.nan_to_num(deltas.values.astype(np.float32), nan=0.0)
        score_values: np.ndarray = np.minimum(angles.scores, deltas.scores).astype(np.float32)

        # Activate shader program
        glUseProgram(self.shader_program)

        # Configure shader uniforms
        glUniform1fv(glGetUniformLocation(self.shader_program, "angles"), len(angle_values), angle_values)
        glUniform1fv(glGetUniformLocation(self.shader_program, "deltas"), len(delta_values), delta_values)
        glUniform1fv(glGetUniformLocation(self.shader_program, "scores"), len(score_values), score_values)
        glUniform1i(glGetUniformLocation(self.shader_program, "num_joints"), len(angles))
        glUniform1f(glGetUniformLocation(self.shader_program, "value_min"), angles.range()[0])
        glUniform1f(glGetUniformLocation(self.shader_program, "value_max"), angles.range()[1])
        glUniform1f(glGetUniformLocation(self.shader_program, "line_thickness"), line_thickness)
        glUniform1f(glGetUniformLocation(self.shader_program, "line_smooth"), line_smooth)
        glUniform4f(glGetUniformLocation(self.shader_program, "color_odd"), *color_odd)
        glUniform4f(glGetUniformLocation(self.shader_program, "color_even"), *color_even)

        # Render
        draw_quad()

        # Cleanup
        glUseProgram(0)

