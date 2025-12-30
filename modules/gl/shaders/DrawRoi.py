from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.utils.PointsAndRects import Rect, Point2f

class DrawRoi(Shader):
    def use(self, fbo, tex0, roi: Rect, rotation_radians: float = 0.0, rotation_center_texture_space: Point2f = Point2f(0.5, 0.5), texture_aspect: float = 1.0, flip_x: bool = False, flip_y: bool = True) -> None:
        if not self.allocated: return
        if not fbo or not tex0: return

        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0)

        # Prepare flipped ROI coordinates
        roi_x = roi.x + roi.width if flip_x else roi.x
        roi_y = roi.y + roi.height if flip_y else roi.y
        roi_w = -roi.width if flip_x else roi.width
        roi_h = -roi.height if flip_y else roi.height

        shader = self.shader_program
        glUseProgram(shader)
        glUniform1i(glGetUniformLocation(shader, "tex0"), 0)
        glUniform4f(glGetUniformLocation(shader, "roi"), roi_x, roi_y, roi_w, roi_h)
        glUniform1f(glGetUniformLocation(shader, "rotation"), rotation_radians)
        glUniform2f(glGetUniformLocation(shader, "rotationCenter"), *rotation_center_texture_space)
        glUniform1f(glGetUniformLocation(shader, "aspectRatio"), texture_aspect)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        draw_quad()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        glUseProgram(0)

        glBindTexture(GL_TEXTURE_2D, 0)

