import numpy as np

from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad

from modules.pose.features import Points2D

class PosePointLines(Shader):
    def use(self, fbo, points: Points2D, line_width: float = 0.01, line_smooth: float = 0.01,
            color: tuple[float, float, float, float] | None = None, use_scores: bool = True) -> None:
        if not self.allocated: return
        if not fbo or not points: return

        # Pack data: [x, y, score, visibility] per point
        n_points: int = len(points.values)
        packed_data: np.ndarray = np.zeros((n_points, 4), dtype=np.float32)
        packed_data[:, 0:2] = np.nan_to_num(points.values, nan=-1.0)
        packed_data[:, 1] = 1.0 - np.nan_to_num(points.values[:, 1], nan=-1.0)  # Flip Y
        packed_data[:, 2] = points.scores if use_scores else np.ones(n_points, dtype=np.float32)
        packed_data[:, 3] = (~np.isnan(points.values[:, 0])).astype(np.float32)

        # Set uniforms
        s = self.shader_program
        glUseProgram(s)

        # Get FBO dimensions for aspect ratio
        viewport = glGetIntegerv(GL_VIEWPORT)
        aspect_ratio = viewport[2] / viewport[3] if viewport[3] > 0 else 1.0

        glUniform1f(glGetUniformLocation(s, "line_width"), line_width)
        glUniform1f(glGetUniformLocation(s, "line_smooth"), line_smooth)
        glUniform1f(glGetUniformLocation(s, "aspect_ratio"), aspect_ratio)

        # Set line_color: use provided color or sentinel value for default colors
        if color is not None:
            glUniform4f(glGetUniformLocation(s, "line_color"), *color)
        else:
            glUniform4f(glGetUniformLocation(s, "line_color"), -1.0, -1.0, -1.0, -1.0)

        # Upload points array
        points_loc = glGetUniformLocation(s, "points")
        glUniform4fv(points_loc, n_points, packed_data.flatten())

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

