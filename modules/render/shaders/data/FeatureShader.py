from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

import numpy as np

from modules.pose.features import PoseFeatureType as PoseFeatureUnion

class FeatureShader(Shader):
    def use(self, feature: PoseFeatureUnion, colors: list[tuple[float, float, float, float]],
            line_width: float = 0.1, line_smooth: float = 0.01, use_scores: bool = False,
            display_range: tuple[float, float] | None = None) -> None:
        if not self.allocated or not self.shader_program:
            print("FeatureBand shader not allocated or shader program missing.")
            return

        # Flatten values and scores to pass to shader
        values: np.ndarray = np.nan_to_num(feature.values.astype(np.float32), nan=0.0)
        scores: np.ndarray = feature.scores.astype(np.float32)

        # Use provided display_range or get from feature
        if display_range is None:
            display_range = feature.display_range()

        # Convert colors to flat array
        colors_array = np.array(colors, dtype=np.float32).flatten()

        # Activate shader program
        glUseProgram(self.shader_program)

        # Configure shader uniforms
        glUniform1fv(self.get_uniform_loc("values"), len(values), values)
        glUniform1fv(self.get_uniform_loc("scores"), len(scores), scores)
        glUniform4fv(self.get_uniform_loc("colors"), len(colors), colors_array)
        glUniform1i(self.get_uniform_loc("num_joints"), len(feature))
        glUniform1i(self.get_uniform_loc("num_colors"), len(colors))
        glUniform2f(self.get_uniform_loc("display_range"), display_range[0], display_range[1])
        glUniform1f(self.get_uniform_loc("line_width"), line_width)
        glUniform1f(self.get_uniform_loc("line_smooth"), line_smooth)
        glUniform1i(self.get_uniform_loc("use_scores"), 1 if use_scores else 0)

        # Render
        draw_quad()
