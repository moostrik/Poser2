from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class PoseAngleWindow(Shader):
    def use(
        self,
        tex0: Texture,
        num_samples: int,
        num_streams: int,
        line_width: float,
        output_aspect_ratio: float = 1.0,
        display_range: float = 3.14159,
    ) -> None:
        if not self.allocated or not self.shader_program:
            print("StreamPose shader not allocated or shader program missing.")
            return
        if not tex0.allocated:
            print("StreamPose shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform1f(self.get_uniform_loc("sample_step"), 1.0 / num_samples)
        glUniform1i(self.get_uniform_loc("num_streams"), num_streams)
        glUniform1f(self.get_uniform_loc("stream_step"), 1.0 / num_streams)
        glUniform1f(self.get_uniform_loc("line_width"), line_width)
        glUniform1f(self.get_uniform_loc("output_aspect_ratio"), output_aspect_ratio)
        glUniform1f(self.get_uniform_loc("display_range"), display_range)

        draw_quad()
