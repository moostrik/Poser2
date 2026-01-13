from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture
import numpy as np


class StreamCorrelation(Shader):
    # Channel encoding for stream textures
    SIMILARITY_CHANNEL = 0  # Similarity values
    MASK_CHANNEL = 1        # Valid mask (1.0 = valid, 0.0 = NaN)

    def use(self, tex0: Texture, num_samples: int, num_streams: int, line_width: float) -> None:
        """Render correlation streams using shader.

        Args:
            tex0: Source texture object containing stream data
            num_samples: Width of texture (capacity)
            num_streams: Height of texture (number of pairs)
            line_width: Line width in normalized coordinates
        """

        if not self.allocated or not self.shader_program:
            print("StreamCorrelation shader not allocated or shader program missing.")
            return
        if not tex0.allocated:
            print("StreamCorrelation shader: input texture not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)

        # Configure shader uniforms
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)
        glUniform1f(glGetUniformLocation(self.shader_program, "sample_step"), 1.0 / num_samples)
        glUniform1i(glGetUniformLocation(self.shader_program, "num_streams"), num_streams)
        glUniform1f(glGetUniformLocation(self.shader_program, "stream_step"), 1.0 / num_streams)
        glUniform1f(glGetUniformLocation(self.shader_program, "line_width"), line_width)

        # Render
        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)

    @staticmethod
    def r_stream_to_image(pair_arrays: list[np.ndarray], num_streams: int) -> np.ndarray:
        """Convert similarity arrays to RGB image with mask - OPTIMIZED.

        Args:
            pair_arrays: List of similarity arrays (one per pair), must not be empty
            num_streams: Number of streams (height of output image)

        Returns:
            RGB image array (num_streams, capacity, 3) where:
            - Channel 0 (R): Unused (zeros)
            - Channel 1 (G): Valid data mask (1.0 = valid, 0.0 = NaN)
            - Channel 2 (B): Similarity values

        Raises:
            ValueError: If pair_arrays is empty
        """
        if not pair_arrays:
            raise ValueError("pair_arrays must not be empty")

        capacity: int = pair_arrays[0].shape[0]
        image: np.ndarray = np.zeros((num_streams, capacity, 3), dtype=np.float32)

        for i, stream in enumerate(pair_arrays[:num_streams]):
            valid_mask: np.ndarray = ~np.isnan(stream)
            image[i, :, StreamCorrelation.SIMILARITY_CHANNEL] = np.nan_to_num(stream, nan=0.0)  # Blue: similarity values
            image[i, valid_mask, StreamCorrelation.MASK_CHANNEL] = 1.0                     # Green: valid mask

        return image
