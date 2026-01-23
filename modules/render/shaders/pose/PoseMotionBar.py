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
        glUniform1fv(self.get_uniform_loc("values"), len(values), values)
        glUniform1fv(self.get_uniform_loc("scores"), len(scores), scores)
        glUniform1i(self.get_uniform_loc("num_joints"), len(feature))
        glUniform1f(self.get_uniform_loc("value_min"), min_range)
        glUniform1f(self.get_uniform_loc("value_max"), max_range)
        glUniform1f(self.get_uniform_loc("line_thickness"), line_thickness)
        glUniform1f(self.get_uniform_loc("line_smooth"), line_smooth)
        glUniform4f(self.get_uniform_loc("color"), *color)
        glUniform4f(self.get_uniform_loc("color_odd"), *color_odd)
        glUniform4f(self.get_uniform_loc("color_even"), *color_even)

        # Render
        draw_quad()

        # Cleanup
        glUseProgram(0)

