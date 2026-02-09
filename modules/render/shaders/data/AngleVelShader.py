from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

import numpy as np

from modules.pose.features import Angles, AngleVelocity

class AngleVelShader(Shader):
    def use(self, angles: Angles, deltas: AngleVelocity, line_width: float = 0.1, line_smooth: float = 0.01,
            colors: list[tuple[float, float, float, float]] = [(1.0, 0.5, 0.0, 1.0), (0.0, 1.0, 1.0, 1.0)],
            display_range: tuple[float, float] | None = None) -> None:
        if not self.allocated or not self.shader_program:
            print("PoseAngleDeltaBar shader not allocated or shader program missing.")
            return

        # Flatten values and scores to pass to shader
        angle_values: np.ndarray = np.nan_to_num(angles.values.astype(np.float32), nan=0.0)
        delta_values: np.ndarray = np.nan_to_num(deltas.values.astype(np.float32), nan=0.0)
        score_values: np.ndarray = np.minimum(angles.scores, deltas.scores).astype(np.float32)

        # Use provided display_range or get from angles
        if display_range is None:
            display_range = angles.display_range()

        # Extract odd/even colors from list
        color_odd = colors[0] if len(colors) > 0 else (1.0, 0.5, 0.0, 1.0)
        color_even = colors[1] if len(colors) > 1 else color_odd

        # Activate shader program
        glUseProgram(self.shader_program)

        # Configure shader uniforms
        glUniform1fv(self.get_uniform_loc("angles"), len(angle_values), angle_values)
        glUniform1fv(self.get_uniform_loc("deltas"), len(delta_values), delta_values)
        glUniform1fv(self.get_uniform_loc("scores"), len(score_values), score_values)
        glUniform1i(self.get_uniform_loc("num_joints"), len(angles))
        glUniform2f(self.get_uniform_loc("display_range"), display_range[0], display_range[1])
        glUniform1f(self.get_uniform_loc("line_width"), line_width)
        glUniform1f(self.get_uniform_loc("line_smooth"), line_smooth)
        glUniform4f(self.get_uniform_loc("color_odd"), *color_odd)
        glUniform4f(self.get_uniform_loc("color_even"), *color_even)

        # Render
        draw_quad()

