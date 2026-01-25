from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture
from modules.utils.PointsAndRects import Rect, Point2f

class DrawRoi(Shader):
    def use(self, tex0: Texture, roi: Rect, rotation_radians: float = 0.0, rotation_center_texture_space: Point2f = Point2f(0.5, 0.5), texture_aspect: float = 1.0) -> None:
        if not self.allocated or not self.shader_program:
            print("DrawRoi shader not allocated or shader program missing.")
            return
        if not tex0.allocated:
            # print("DrawRoi shader: input texture not allocated.")
            return

        # Activate shader program
        glUseProgram(self.shader_program)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)

        # Configure shader uniforms (using cached locations)
        glUniform1i(self.get_uniform_loc("tex0"), 0)
        glUniform4f(self.get_uniform_loc("roi"), roi.x, roi.y, roi.width, roi.height)
        glUniform1f(self.get_uniform_loc("rotation"), rotation_radians)
        glUniform2f(self.get_uniform_loc("rotationCenter"), *rotation_center_texture_space)
        glUniform1f(self.get_uniform_loc("aspectRatio"), texture_aspect)

        # Render
        draw_quad()
