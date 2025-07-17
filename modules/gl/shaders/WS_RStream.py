from OpenGL.GL import * # type: ignore
from OpenGL.GL.shaders import ShaderProgram # type: ignore
from modules.gl.Shader import Shader, draw_quad

from modules.correlation.PairCorrelationStream import PairCorrelationStreamData
import numpy as np
from typing import Tuple

class WS_RStream(Shader):
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
        pairs: list[Tuple[int, int]] = r_streams.get_top_pairs(num_streams)
        capacity: int = r_streams.capacity

        image: np.ndarray = np.zeros((num_streams, capacity, 3), dtype=np.float32)

        for i in range(num_streams):
            pair: Tuple[int, int] = pairs[i]
            stream: np.ndarray | None = r_streams.get_metric_window(pair)
            if stream is not None:
                steam_len: int = stream.shape[0]
                if steam_len >= capacity:
                    image[i, :, 2] = stream[-capacity:]
                    image[i, :, 1] = 1.0
                else:
                    start_idx: int = capacity - steam_len
                    image[i, start_idx:, 2] = stream
                    image[i, start_idx:, 1] = 1.0

                    image[i, start_idx, 1] = 0.0
                    if steam_len > 1:
                        image[i, start_idx + 1, 1] = 0.0

        return image

    @staticmethod
    def r_stream_to_visible_image(r_streams: PairCorrelationStreamData, num_streams: int) -> np.ndarray:
        pairs: list[Tuple[int, int]] = r_streams.get_top_pairs(num_streams)
        capacity: int = r_streams.capacity

        image: np.ndarray = np.zeros((capacity, num_streams, 4), dtype=np.float32)

        for i in range(num_streams):
            pair: Tuple[int, int] = pairs[i]
            r: np.ndarray | None = r_streams.get_metric_window(pair)
            if r is not None:
                if r.shape[0] <= capacity:
                    image[-r.shape[0]:, i, 0] = r
                    image[-r.shape[0]:, i, 1] = r
                    image[-r.shape[0]:, i, 2] = r
                    image[-r.shape[0]:, i, 3] = 1.0  # Alpha channel (full opacity)
                else:
                    # Take the most recent data if longer than capacity
                    image[:, i, 0] = r[-capacity:]
                    image[:, i, 1] = r[-capacity:]
                    image[:, i, 2] = r[-capacity:]
                    image[:, i, 3] = 1.0  # Alpha channel (full opacity)

        return image.transpose(1, 0, 2)  # Transpose to (num_streams, capacity, 4)