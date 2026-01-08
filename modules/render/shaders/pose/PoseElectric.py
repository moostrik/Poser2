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

    def use(self, fbo, pose: Frame ) -> None:
        if not self.allocated: return
        if not fbo or not pose: return

        points: Points2D = pose.points

        # Pack data: [x, y, score, visibility] per point
        n_points: int = len(points.values)
        packed_data: np.ndarray = np.zeros((n_points, 4), dtype=np.float32)
        packed_data[:, 0:2] = np.nan_to_num(points.values, nan=-1.0)
        packed_data[:, 1] = 1.0 - np.nan_to_num(points.values[:, 1], nan=-1.0)  # Flip Y

        # Set uniforms
        s = self.shader_program
        glUseProgram(s)

        # Get FBO dimensions for resolution uniform
        viewport = glGetIntegerv(GL_VIEWPORT)
        resolution = np.array([viewport[2], viewport[3]], dtype=np.float32)

        # Upload resolution
        resolution_loc = glGetUniformLocation(s, "resolution")
        glUniform2fv(resolution_loc, 1, resolution)

        # Upload time
        current_time = pytime.time() - self.start_time
        time_loc = glGetUniformLocation(s, "time")
        glUniform1f(time_loc, current_time)

        # Upload points array
        points_loc = glGetUniformLocation(s, "points")
        glUniform4fv(points_loc, n_points, packed_data.flatten())

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

