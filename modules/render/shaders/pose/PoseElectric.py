import numpy as np
import time as pytime

from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad_pixels
from modules.gl import Fbo

from modules.pose.Frame import Frame
from modules.pose.features import Points2D

class PoseElectric(Shader):
    def __init__(self) -> None:
        super().__init__()
        self.start_time = pytime.time()

    def use(self, fbo: Fbo, pose: Frame) -> None:
        if not self.allocated or not self.shader_program: return
        if not fbo.allocated or not pose: return

        points: Points2D = pose.points

        # Pack data: [x, y, score, visibility] per point
        n_points: int = len(points.values)
        packed_data: np.ndarray = np.zeros((n_points, 4), dtype=np.float32)
        packed_data[:, 0:2] = np.nan_to_num(points.values, nan=-1.0)
        packed_data[:, 1] = 1.0 - np.nan_to_num(points.values[:, 1], nan=-1.0)  # Flip Y

        # Activate shader program
        glUseProgram(self.shader_program)

        # Set up render target
        glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo_id)
        glViewport(0, 0, fbo.width, fbo.height)

        # Configure shader uniforms
        glUniform2f(glGetUniformLocation(self.shader_program, "resolution"), float(fbo.width), float(fbo.height))
        glUniform1f(glGetUniformLocation(self.shader_program, "time"), pytime.time() - self.start_time)
        glUniform4fv(glGetUniformLocation(self.shader_program, "points"), n_points, packed_data.flatten())

        # Render
        draw_quad_pixels(fbo.width, fbo.height)

        # Cleanup
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)

