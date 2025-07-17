from OpenGL.GL import * # type: ignore
from OpenGL.GL.shaders import ShaderProgram # type: ignore
from modules.gl.Shader import Shader, draw_quad

from modules.pose.PoseStream import PoseStreamData
import numpy as np

class WS_PoseStream(Shader):
    def __init__(self) -> None:
        super().__init__()
        self.shader_name = self.__class__.__name__

    def allocate(self, monitor_file = False) -> None:
        super().allocate(self.shader_name, monitor_file)

    def use(self, fbo: int, tex0: int, width: int, height: int, linewidth: float) -> None :
        super().use()
        if not self.allocated: return
        if not fbo or not tex0: return
        s: ShaderProgram | None = self.shader_program
        if s is None: return

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0)

        glUseProgram(s)
        glUniform1i(glGetUniformLocation(s, "tex0"), 0)
        glUniform1f(glGetUniformLocation(s, "inv_width"),   1.0  / width)
        glUniform1i(glGetUniformLocation(s, "num_joints"),  height)
        glUniform1f(glGetUniformLocation(s, "joint_range"), 1.0  / height)
        glUniform1f(glGetUniformLocation(s, "line_width"),  (linewidth  / height) ** 2)  # line width squared

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

        glBindTexture(GL_TEXTURE_2D, 0)


    @staticmethod
    def pose_stream_to_image(pose_stream: PoseStreamData, confidence_ceil: bool = True) -> np.ndarray:
        angles_raw: np.ndarray = np.nan_to_num(pose_stream.angles.to_numpy(), nan=0.0).astype(np.float32)
        angles_norm: np.ndarray = np.clip(np.abs(angles_raw) / np.pi, 0, 1)
        sign_channel: np.ndarray = (angles_raw > 0).astype(np.float32)
        if confidence_ceil:
            confidences: np.ndarray = np.where(pose_stream.confidences.to_numpy() > 0, 1.0, 0.0).astype(np.float32)
        else:
            confidences: np.ndarray = np.clip(pose_stream.confidences.to_numpy().astype(np.float32), 0, 1)
        # width, height = angles_norm.shape
        # img: np.ndarray = np.ones((height, width, 4), dtype=np.float32)
        # img[..., 2] = angles_norm.T   r
        # img[..., 1] = sign_channel.T  g
        # img[..., 0] = confidences.T   b
        channels: np.ndarray = np.stack([confidences, sign_channel, angles_norm], axis=-1)
        image: np.ndarray = channels.transpose(1, 0, 2)

        # padding
        capacity: int = pose_stream.capacity
        if image.shape[1] > capacity:
            image = image[:capacity, :, :]
        if image.shape[1] < capacity:
            pad_width: int = capacity - image.shape[1]
            # Use first value for padding
            pad_value: np.ndarray = image[:, 0, :].copy() if image.shape[1] > 0 else np.zeros((image.shape[0], image.shape[2]), dtype=image.dtype)
            # Set confidences to zero for all rows
            pad_value[:, 0] = 0.0
            pad: np.ndarray = np.repeat(pad_value[:, np.newaxis, :], pad_width, axis=1)
            image = np.concatenate([pad, image], axis=1)

        return image.astype(np.float32)