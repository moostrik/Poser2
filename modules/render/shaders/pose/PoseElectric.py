import numpy as np
import time as pytime

from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

from modules.pose.Frame import Frame
from modules.pose.features import Points2D

class PoseElectric(Shader):
    def __init__(self) -> None:
        super().__init__()
        self.start_time = pytime.time()

    def use(self, pose: Frame) -> None:
        if not self.allocated or not self.shader_program:
            print("PoseElectric shader not allocated or shader program missing.")
            return
        if not pose:
            print("PoseElectric shader: pose not provided.")
            return

        points: Points2D = pose.points

        # Pack data: [x, y, score, visibility] per point
        n_points: int = len(points.values)
        packed_data: np.ndarray = np.zeros((n_points, 4), dtype=np.float32)
        packed_data[:, 0:2] = np.nan_to_num(points.values, nan=-1.0)
        packed_data[:, 1] = 1.0 - np.nan_to_num(points.values[:, 1], nan=-1.0)  # Flip Y

        # Activate shader program
        glUseProgram(self.shader_program)

        # Configure shader uniforms
        glUniform1f(self.get_uniform_loc("time"), pytime.time() - self.start_time)
        glUniform4fv(self.get_uniform_loc("points"), n_points, packed_data.flatten())

        # Render
        draw_quad()

