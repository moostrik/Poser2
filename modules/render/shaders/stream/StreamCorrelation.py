from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
import numpy as np


class StreamCorrelation(Shader):
    # Channel encoding for stream textures
    SIMILARITY_CHANNEL = 0  # Similarity values
    MASK_CHANNEL = 1        # Valid mask (1.0 = valid, 0.0 = NaN)

    def use(self, fbo: int, tex0: int, num_samples: int, num_streams: int, line_width: float) -> None:
        """Render correlation streams using shader.

        Args:
            fbo: Target framebuffer ID (0 = default framebuffer)
            tex0: Source texture ID containing stream data
            num_samples: Width of texture (capacity)
            num_streams: Height of texture (number of pairs)
            line_width: Line width in normalized coordinates
        """

        # Guard clauses
        if not self.allocated:
            return
        if not tex0:
            return
        if self.shader_program is None:
            return

        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0)

        # Setup shader
        glUseProgram(self.shader_program)
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)
        glUniform1f(glGetUniformLocation(self.shader_program, "sample_step"), 1.0 / num_samples)
        glUniform1i(glGetUniformLocation(self.shader_program, "num_streams"), num_streams)
        glUniform1f(glGetUniformLocation(self.shader_program, "stream_step"), 1.0 / num_streams)
        glUniform1f(glGetUniformLocation(self.shader_program, "line_width"), line_width)

        # Render to FBO
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Cleanup
        glUseProgram(0)
        glBindTexture(GL_TEXTURE_2D, 0)

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
