from OpenGL.GL import * # type: ignore
from modules.gl.Shader import Shader, draw_quad_pixels
from modules.gl import Fbo, Texture
from modules.utils.PointsAndRects import Rect, Point2f

class DrawRoi(Shader):
    def use(self, fbo: Fbo, tex0: Texture, roi: Rect, rotation_radians: float = 0.0, rotation_center_texture_space: Point2f = Point2f(0.5, 0.5), texture_aspect: float = 1.0, flip_x: bool = False, flip_y: bool = True) -> None:
        if not self.allocated or not self.shader_program: return
        if not fbo.allocated or not tex0.allocated: return

        # Prepare flipped ROI coordinates
        roi_x = roi.x + roi.width if flip_x else roi.x
        roi_y = roi.y + roi.height if flip_y else roi.y
        roi_w = -roi.width if flip_x else roi.width
        roi_h = -roi.height if flip_y else roi.height

        # Activate shader program
        glUseProgram(self.shader_program)

        # Set up render target
        glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo_id)
        glViewport(0, 0, fbo.width, fbo.height)

        # Bind input texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex0.tex_id)

        # Configure shader uniforms
        glUniform2f(glGetUniformLocation(self.shader_program, "resolution"), float(fbo.width), float(fbo.height))
        glUniform1i(glGetUniformLocation(self.shader_program, "tex0"), 0)
        glUniform4f(glGetUniformLocation(self.shader_program, "roi"), roi_x, roi_y, roi_w, roi_h)
        glUniform1f(glGetUniformLocation(self.shader_program, "rotation"), rotation_radians)
        glUniform2f(glGetUniformLocation(self.shader_program, "rotationCenter"), *rotation_center_texture_space)
        glUniform1f(glGetUniformLocation(self.shader_program, "aspectRatio"), texture_aspect)

        # Render
        draw_quad_pixels(fbo.width, fbo.height)

        # Cleanup
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glUseProgram(0)

