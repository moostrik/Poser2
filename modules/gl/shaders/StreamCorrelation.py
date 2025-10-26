from OpenGL.GL import * # type: ignore
from OpenGL.GL.shaders import ShaderProgram # type: ignore
from modules.gl.Shader import Shader, draw_quad

from modules.pose.correlation.PairCorrelationStream import PairCorrelationStreamData
import numpy as np
from typing import Tuple

class StreamCorrelation(Shader):
    def __init__(self) -> None:
        super().__init__()
        self.shader_name = self.__class__.__name__

    def allocate(self, monitor_file = False) -> None:
        super().allocate(self.shader_name, monitor_file)

    def use(self, fbo: int, tex0: int, num_samples: int, num_streams: int, line_width: float) -> None :
        super().use()
        if not self.allocated: return
        if not fbo or not tex0: return
        s: ShaderProgram | None = self.shader_program
        if s is None: return

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0)

        glUseProgram(s)
        glUniform1i(glGetUniformLocation(s, "tex0"), 0)
        glUniform1f(glGetUniformLocation(s, "sample_step"), 1.0  / num_samples)
        glUniform1i(glGetUniformLocation(s, "num_streams"), num_streams)
        glUniform1f(glGetUniformLocation(s, "stream_step"), 1.0  / num_streams)
        glUniform1f(glGetUniformLocation(s, "line_width"),  line_width)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

        glBindTexture(GL_TEXTURE_2D, 0)


    @staticmethod
    def r_stream_to_image(r_streams: PairCorrelationStreamData, num_streams: int) -> np.ndarray:
        """Convert stream data to RGB image with mask - OPTIMIZED."""
        capacity: int = r_streams.capacity

        # ✅ Get pairs and windows in one call
        pairs_with_windows = r_streams.get_top_pairs_with_windows(num_streams, capacity, "similarity")

        image: np.ndarray = np.zeros((num_streams, capacity, 3), dtype=np.float32)

        for i, (pair_id, stream) in enumerate(pairs_with_windows):
            stream_len: int = stream.shape[0]

            if stream_len >= capacity:
                data = stream[-capacity:]
            else:
                data = np.zeros(capacity, dtype=np.float32)
                start_idx: int = capacity - stream_len
                data[start_idx:] = stream

            valid_mask = ~np.isnan(data)

            image[i, :, 2] = np.nan_to_num(data, nan=0.0)
            image[i, valid_mask, 1] = 1.0
            image[i, ~valid_mask, 1] = 0.0

            if stream_len < capacity:
                start_idx = capacity - stream_len
                image[i, start_idx, 0] = 1.0
                if stream_len > 1:
                    image[i, start_idx + 1, 0] = 1.0

        return image

    @staticmethod
    def r_stream_to_visible_image(r_streams: PairCorrelationStreamData, num_streams: int) -> np.ndarray:
        """Convert stream data to visible RGB image with mask.

        Returns:
            RGB image array (num_streams, capacity, 3) where:
            - Channel 0 (R): Similarity value (for visualization)
            - Channel 1 (G): Similarity value (for visualization)
            - Channel 2 (B): Similarity value (for visualization)
            - Alpha in separate channel 3: 1.0 = valid, 0.3 = NaN gap
        """
        pairs: list[Tuple[int, int]] = r_streams.get_top_pairs(num_streams)
        num_pairs: int = len(pairs)
        capacity: int = r_streams.capacity

        image: np.ndarray = np.zeros((capacity, num_streams, 4), dtype=np.float32)

        for i in range(num_pairs):
            pair: Tuple[int, int] = pairs[i]
            r: np.ndarray | None = r_streams.get_metric_window_array(pair)

            if r is not None:
                if r.shape[0] <= capacity:
                    data = r
                    start_idx = capacity - r.shape[0]
                else:
                    # Take the most recent data if longer than capacity
                    data = r[-capacity:]
                    start_idx = 0

                # ✅ Create validity mask
                valid_mask = ~np.isnan(data)

                # ✅ Replace NaN with gray value for visualization
                data_clean = np.nan_to_num(data, nan=0.5)

                if start_idx > 0:
                    # Pad at beginning
                    image[start_idx:, i, 0] = data_clean  # R
                    image[start_idx:, i, 1] = data_clean  # G
                    image[start_idx:, i, 2] = data_clean  # B

                    # ✅ Alpha channel: 1.0 for valid, 0.3 for NaN
                    image[start_idx:, i, 3][valid_mask] = 1.0
                    image[start_idx:, i, 3][~valid_mask] = 0.3
                else:
                    # Full capacity
                    image[:, i, 0] = data_clean  # R
                    image[:, i, 1] = data_clean  # G
                    image[:, i, 2] = data_clean  # B

                    # ✅ Alpha channel: 1.0 for valid, 0.3 for NaN
                    image[:, i, 3][valid_mask] = 1.0
                    image[:, i, 3][~valid_mask] = 0.3

        return image.transpose(1, 0, 2)  # Transpose to (num_streams, capacity, 4)