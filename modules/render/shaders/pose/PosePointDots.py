import numpy as np

from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

from modules.pose.features import Points2D

class PosePointDots(Shader):
    def use(self, points: Points2D, dot_size: float = 0.01, dot_smooth: float = 0.01) -> None:
        if not self.allocated or not self.shader_program:
            print("PosePointDots shader not allocated or shader program missing.")
            return
        if not points:
            print("PosePointDots shader: points not provided.")
            return

        # Pack data: [x, y, score, visibility] per point
        n_points: int = len(points.values)
        packed_data: np.ndarray = np.zeros((n_points, 4), dtype=np.float32)
        packed_data[:, 0:2] = np.nan_to_num(points.values, nan=-1.0)
        packed_data[:, 1] = 1.0 - np.nan_to_num(points.values[:, 1], nan=-1.0)  # Flip Y
        packed_data[:, 2] = points.scores
        packed_data[:, 3] = (~np.isnan(points.values[:, 0])).astype(np.float32)

        # Activate shader program
        glUseProgram(self.shader_program)

        # Configure shader uniforms
        glUniform1i(self.get_uniform_loc("num_points"), n_points)
        glUniform1f(self.get_uniform_loc("dot_size"), dot_size)
        glUniform1f(self.get_uniform_loc("dot_smooth"), dot_smooth)
        glUniform4fv(self.get_uniform_loc("points"), n_points, packed_data.flatten())

        # Render
        draw_quad()

