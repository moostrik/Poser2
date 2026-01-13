import numpy as np

from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Fbo

from modules.pose.features import Points2D

class PosePointLines(Shader):
    def use(self, fbo: Fbo, points: Points2D, line_width: float = 0.01, line_smooth: float = 0.01,
            color: tuple[float, float, float, float] | None = None, use_scores: bool = True) -> None:
        if not self.allocated or not self.shader_program: return
        if not fbo.allocated or not points: return

        # Pack data: [x, y, score, visibility] per point
        n_points: int = len(points.values)
        packed_data: np.ndarray = np.zeros((n_points, 4), dtype=np.float32)
        packed_data[:, 0:2] = np.nan_to_num(points.values, nan=-1.0)
        packed_data[:, 1] = 1.0 - np.nan_to_num(points.values[:, 1], nan=-1.0)  # Flip Y
        packed_data[:, 2] = points.scores if use_scores else np.ones(n_points, dtype=np.float32)
        packed_data[:, 3] = (~np.isnan(points.values[:, 0])).astype(np.float32)

        # Activate shader program
        glUseProgram(self.shader_program)

        # Set up render target
        glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo_id)
        glViewport(0, 0, fbo.width, fbo.height)

        # Calculate aspect ratio
        aspect_ratio = fbo.width / fbo.height if fbo.height > 0 else 1.0

        # Configure shader uniforms
        glUniform2f(glGetUniformLocation(self.shader_program, "resolution"), float(fbo.width), float(fbo.height))
        glUniform1f(glGetUniformLocation(self.shader_program, "line_width"), line_width)
        glUniform1f(glGetUniformLocation(self.shader_program, "line_smooth"), line_smooth)
        glUniform1f(glGetUniformLocation(self.shader_program, "aspect_ratio"), aspect_ratio)
        if color is not None:
            glUniform4f(glGetUniformLocation(self.shader_program, "line_color"), *color)
        else:
            glUniform4f(glGetUniformLocation(self.shader_program, "line_color"), -1.0, -1.0, -1.0, -1.0)
        glUniform4fv(glGetUniformLocation(self.shader_program, "points"), n_points, packed_data.flatten())

        # Render
        draw_quad()

        # Cleanup
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)

