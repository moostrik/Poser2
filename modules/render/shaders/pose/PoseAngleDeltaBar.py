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
        glUniform1fv(self.get_uniform_loc("angles"), len(angle_values), angle_values)
        glUniform1fv(self.get_uniform_loc("deltas"), len(delta_values), delta_values)
        glUniform1fv(self.get_uniform_loc("scores"), len(score_values), score_values)
        glUniform1i(self.get_uniform_loc("num_joints"), len(angles))
        glUniform1f(self.get_uniform_loc("value_min"), angles.range()[0])
        glUniform1f(self.get_uniform_loc("value_max"), angles.range()[1])
        glUniform1f(self.get_uniform_loc("line_thickness"), line_thickness)
        glUniform1f(self.get_uniform_loc("line_smooth"), line_smooth)
        glUniform4f(self.get_uniform_loc("color_odd"), *color_odd)
        glUniform4f(self.get_uniform_loc("color_even"), *color_even)

        # Render
        draw_quad()

