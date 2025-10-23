from OpenGL.GL import * # type: ignore
from OpenGL.GL.shaders import ShaderProgram # type: ignore
from modules.gl.Shader import Shader, draw_quad

from modules.pose.PoseStream import PoseStreamData
import numpy as np

class StreamPose(Shader):
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
    def pose_stream_to_image(pose_stream: PoseStreamData, confidence_ceil: bool = False) -> np.ndarray:

        angles_raw: np.ndarray = np.nan_to_num(pose_stream.angles.to_numpy(), nan=0.0).astype(np.float32)
        confidences_raw: np.ndarray = pose_stream.confidences.to_numpy()

        angles_norm: np.ndarray = np.clip(np.abs(angles_raw) / np.pi, 0, 1)
        sign_channel: np.ndarray = (angles_raw > 0).astype(np.float32)
        if confidence_ceil:
            confidences: np.ndarray = (confidences_raw > 0).astype(np.float32)
        else:
            confidences: np.ndarray = np.clip(confidences_raw.astype(np.float32), 0, 1)

        try:
            streams: np.ndarray = np.stack([confidences, sign_channel, angles_norm], axis=-1).transpose(1, 0, 2)
        except Exception as e:
            print(len(confidences), len(sign_channel), len(angles_norm))
            print(e)
            return np.zeros((3, pose_stream.capacity, 3), dtype=np.float32)

        capacity: int = pose_stream.capacity
        stream_len: int = streams.shape[1]

        # Pre-allocate the final image with the target capacity
        image: np.ndarray = np.zeros((streams.shape[0], capacity, streams.shape[2]), dtype=np.float32)

        if stream_len > 0:
            if stream_len >= capacity:
                # Take the most recent data (right-most columns)
                image[:, :, :] = streams[:, -capacity:, :]
            else:
                # Place data at the end (most recent position)
                start_idx: int = capacity - stream_len
                image[:, start_idx:, :] = streams

                # set the first two confidences to 0.0 for visualization
                image[:, start_idx, 0] = 0.0
                if stream_len > 1:
                    image[:, start_idx + 1, 0] = 0.0

        return image