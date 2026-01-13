from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

import numpy as np

from modules.pose.features import PoseFeatureType as PoseFeatureUnion

class PoseValuesBar(Shader):
    def use(self, norm_values: np.ndarray, colors: np.ndarray, line_thickness: float = 0.1, line_smooth: float = 0.01) -> None:
        if not self.allocated or not self.shader_program:
            print("PoseValuesBar shader not allocated or shader program missing.")
            return

        # Flatten values and colors to pass to shader
        values: np.ndarray = np.nan_to_num(norm_values.astype(np.float32), nan=0.0)
        colors_flat: np.ndarray = colors.astype(np.float32)

        # Activate shader program
        glUseProgram(self.shader_program)

        # Configure shader uniforms
        glUniform1fv(glGetUniformLocation(self.shader_program, "values"), len(values), values)
        glUniform4fv(glGetUniformLocation(self.shader_program, "colors"), len(colors_flat), colors_flat.flatten())
        glUniform1i(glGetUniformLocation(self.shader_program, "num_values"), len(values))
        glUniform1f(glGetUniformLocation(self.shader_program, "line_thickness"), line_thickness)
        glUniform1f(glGetUniformLocation(self.shader_program, "line_smooth"), line_smooth)

        # Render
        draw_quad()

        # Cleanup
        glUseProgram(0)
