from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class WindowShader(Shader):
    """Base class for feature window visualization shaders.

    Provides common interface for rendering temporal feature windows as horizontal time series.
    All window shaders expect GL_RG32F texture input where:
    - R channel = feature value
    - G channel = validity mask (0.0 or 1.0)

    Standard uniforms:
    - tex0: Input texture (feature_len x time x RG)
    - sample_step: 1.0 / num_samples
    - num_streams: Number of feature elements (height)
    - stream_step: 1.0 / num_streams
    - line_width: Line thickness in normalized coordinates
    - output_aspect_ratio: width/height for aspect correction
    - display_range: Max absolute value for normalization
    """

    def use(
        self,
        tex0: Texture,
        num_samples: int,
        num_streams: int,
        line_width: float,
        output_aspect_ratio: float = 1.0,
        display_range: tuple[float, float] = (-3.14159, 3.14159),
        color_even: tuple[float, float, float] = (1.0, 0.5, 0.0),
        color_odd: tuple[float, float, float] = (0.0, 1.0, 1.0),
        alpha: float = 0.5,
    ) -> None:
        """Render feature window visualization.

        Args:
            tex0: Input RG32F texture (feature_len, time, 2)
            num_samples: Number of time samples (width)
            num_streams: Number of feature elements (height)
            line_width: Line thickness in normalized coordinates
            output_aspect_ratio: Output buffer aspect ratio (width/height)
            display_range: (min, max) value range for normalization
            color_even: RGB color for even-numbered streams (default orange)
            color_odd: RGB color for odd-numbered streams (default cyan)
            alpha: Alpha transparency (default 0.5)
        """
        if not self.allocated or not self.shader_program:
            print(f"{self.__class__.__name__} shader not allocated or shader program missing.")
            return
        if not tex0.allocated:
            print(f"{self.__class__.__name__} shader: input texture not allocated.")
            return

        glUseProgram(self.shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)

        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform2f(self.get_uniform_loc("tex_size"), float(tex0.width), float(tex0.height))
        glUniform1f(self.get_uniform_loc("sample_step"), 1.0 / num_samples)
        glUniform1i(self.get_uniform_loc("num_streams"), num_streams)
        glUniform1f(self.get_uniform_loc("stream_step"), 1.0 / num_streams)
        glUniform1f(self.get_uniform_loc("line_width"), line_width)
        glUniform1f(self.get_uniform_loc("output_aspect_ratio"), output_aspect_ratio)
        glUniform1f(self.get_uniform_loc("display_range_min"), display_range[0])
        glUniform1f(self.get_uniform_loc("display_range_max"), display_range[1])
        glUniform3f(self.get_uniform_loc("color_even"), *color_even)
        glUniform3f(self.get_uniform_loc("color_odd"), *color_odd)
        glUniform1f(self.get_uniform_loc("alpha"), alpha)

        draw_quad()
