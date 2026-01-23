"""ObstacleOffset shader - Precompute neighbor obstacle flags."""

from OpenGL.GL import *  # type: ignore
from modules.gl.Shader import Shader, draw_quad
from modules.gl import Texture


class ObstacleOffset(Shader):
    """Precompute neighbor obstacle information for boundary conditions."""

    def __init__(self) -> None:
        super().__init__()

    def use(self, obstacle: Texture) -> None:
        """Compute neighbor obstacle flags.

        Args:
            obstacle: Obstacle mask (R8: 1.0 = obstacle, 0.0 = fluid)

        Output:
            RGBA8 texture where R=top, G=bottom, B=right, A=left neighbor flags
        """
        # Bind shader program
        glUseProgram(self.shader_program)

        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, obstacle.tex_id)
        glUniform1i(self.get_uniform_loc("uObstacle"), 0)

        # Draw fullscreen quad
        draw_quad()

        # Cleanup
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)
