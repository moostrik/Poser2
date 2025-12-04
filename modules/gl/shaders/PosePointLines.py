import numpy as np

from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

from modules.pose.features import Points2D

class PosePointLines(Shader):
    def __init__(self) -> None:
        super().__init__()
        self.shader_name = self.__class__.__name__

    def allocate(self, monitor_file = False) -> None:
        super().allocate(self.shader_name, monitor_file)

    def deallocate(self):
        return super().deallocate()

    def use(self, fbo, points: Points2D, line_width: float = 0.01, line_smooth: float = 0.01) -> None:
        super().use()
        if not self.allocated: return
        if not fbo or not points: return

        # Pack data: [x, y, score, visibility] per point
        n_points: int = len(points.values)
        packed_data: np.ndarray = np.zeros((n_points, 4), dtype=np.float32)
        packed_data[:, 0:2] = np.nan_to_num(points.values, nan=-1.0)
        packed_data[:, 1] = 1.0 - np.nan_to_num(points.values[:, 1], nan=-1.0)  # Flip Y
        packed_data[:, 2] = points.scores
        packed_data[:, 3] = (~np.isnan(points.values[:, 0])).astype(np.float32)

        # Line segment indices
        line_segments = np.array([
            [0, 1], [0, 2], [1, 3], [2, 4],  # Face
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # Arms
            [5, 11], [6, 12], [11, 12],  # Torso
            [11, 13], [13, 15], [12, 14], [14, 16]  # Legs
        ], dtype=np.int32)
        n_segments = len(line_segments)

        # Set uniforms
        s = self.shader_program
        glUseProgram(s)

        # Get FBO dimensions for aspect ratio
        viewport = glGetIntegerv(GL_VIEWPORT)
        aspect_ratio = viewport[2] / viewport[3] if viewport[3] > 0 else 1.0

        glUniform1i(glGetUniformLocation(s, "num_points"), n_points)
        glUniform1i(glGetUniformLocation(s, "num_segments"), n_segments)
        glUniform1f(glGetUniformLocation(s, "line_width"), line_width)
        glUniform1f(glGetUniformLocation(s, "line_smooth"), line_smooth)
        glUniform1f(glGetUniformLocation(s, "aspect_ratio"), aspect_ratio)

        # Upload points and segments as uniform arrays
        points_loc = glGetUniformLocation(s, "points")
        glUniform4fv(points_loc, n_points, packed_data.flatten())

        segments_loc = glGetUniformLocation(s, "segments")
        glUniform2iv(segments_loc, n_segments, line_segments.flatten())

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

