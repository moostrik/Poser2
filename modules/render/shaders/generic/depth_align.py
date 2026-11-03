import numpy as np
from OpenGL.GL import GL_R32F, GL_R32UI, GL_RED_INTEGER, GL_TRUE, GL_UNSIGNED_INT, glClearTexImage, glUniform1i, glUniform2i, glUniformMatrix3fv, \
    glUniformMatrix4x3fv, glUseProgram
from modules.gl import ComputeShader, Texture

IDENTITY: np.ndarray = np.eye(3, dtype=np.float32)
EMPTY: np.ndarray = np.array([0xFFFFFFFF], dtype=np.uint32)


class DepthAlign(ComputeShader):
    """Depth on the video frame's view: moved there point by point, straightened and mirrored, the
    nearest kept where two land on one pixel and 0 where nothing lands."""

    WORKGROUP_SIZE_X = 16
    WORKGROUP_SIZE_Y = 16

    @staticmethod
    def clear(nearest: Texture) -> None:
        """Empty the minimum once after allocating it; each `use` leaves it empty again."""
        glClearTexImage(nearest.tex_id, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, EMPTY)

    def use(self, depth: Texture, nearest: Texture, aligned: Texture, to_ray: np.ndarray, to_uv: np.ndarray,
            homography: np.ndarray | None, flip_h: bool) -> None:
        """`depth` (metres, R32F) onto `aligned` (R32F) through `nearest` (R32UI, empty), by
        `geometry.depth_projection`'s `to_ray` and `to_uv`, then `homography` (None for a level frame)
        and `flip_h`."""
        if not self.allocated or not self.shader_program or not depth.allocated:
            return

        to_corrected: np.ndarray = IDENTITY if homography is None else homography.astype(np.float32)

        glUseProgram(self.shader_program)
        self.bind_texture(0, depth, "depth")
        self.bind_image_readwrite(0, nearest, GL_R32UI)
        self.bind_image_write(1, aligned, GL_R32F)
        glUniformMatrix3fv(self.get_uniform_loc("toRay"), 1, GL_TRUE, np.ascontiguousarray(to_ray, dtype=np.float32))
        glUniformMatrix4x3fv(self.get_uniform_loc("toUv"), 1, GL_TRUE, np.ascontiguousarray(to_uv, dtype=np.float32))
        glUniformMatrix3fv(self.get_uniform_loc("toCorrected"), 1, GL_TRUE, to_corrected)
        glUniform1i(self.get_uniform_loc("flipH"), int(flip_h))
        glUniform2i(self.get_uniform_loc("depthSize"), depth.width, depth.height)
        glUniform2i(self.get_uniform_loc("alignedSize"), aligned.width, aligned.height)

        glUniform1i(self.get_uniform_loc("stage"), 0)
        self.dispatch(depth.width, depth.height)
        glUniform1i(self.get_uniform_loc("stage"), 1)
        self.dispatch(aligned.width, aligned.height)
        self.barrier_texture_fetch()
