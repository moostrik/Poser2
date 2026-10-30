import numpy as np
from OpenGL.GL import GL_TEXTURE0, GL_TEXTURE_2D, GL_TRUE, glActiveTexture, glBindTexture, glUniform1i, glUniformMatrix3fv, glUseProgram
from modules.gl import Shader, draw_quad, Texture

IDENTITY: np.ndarray = np.eye(3, dtype=np.float32)


class DepthWarp(Shader):
    def use(self, depth: Texture, homography: np.ndarray | None, flip_h: bool) -> None:
        """Depth in metres straightened by `homography` (`geometry.tilt_homography`, None for a level
        frame), then mirrored with `flip_h`."""
        if not self.allocated or not self.shader_program or not depth.allocated:
            return

        to_corrected: np.ndarray = IDENTITY if homography is None else homography.astype(np.float32)

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, depth.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("depth"), 0)
        glUniformMatrix3fv(self.get_uniform_loc("toCorrected"), 1, GL_TRUE, to_corrected)
        glUniformMatrix3fv(self.get_uniform_loc("toSource"), 1, GL_TRUE, np.linalg.inv(to_corrected))
        glUniform1i(self.get_uniform_loc("flipH"), int(flip_h))

        # Render
        draw_quad()
