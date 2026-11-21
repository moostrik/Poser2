"""DepthColormap: a depth in meters as a picture, through a colormap or grey, or its confidence as grey or as the
colormap's brightness."""

from enum import IntEnum

from OpenGL.GL import glUniform1f, glUniform1i, glUseProgram
from modules.gl import Colormap, Shader, draw_quad, Texture


class DepthDrawMode(IntEnum):
    """How DepthColormap draws a depth."""
    GREY = 0            # near white, far black
    COLOR = 1           # through the colormap
    OVER_IMAGE = 2      # through the colormap, over the image
    CONFIDENCE = 3      # its confidence, green, as grey from 0 black to 1 white
    COLOR_BY_CONFIDENCE = 4     # through the colormap, as bright as its confidence


class DepthColormap(Shader):
    def use(self, depth: Texture, image: Texture, near: float, far: float, mode: DepthDrawMode, colormap: Colormap = Colormap.JET) -> None:
        """Depth in meters as a picture, drawn as `mode` says, colored by `colormap`; `image` is drawn under it in
        OVER_IMAGE. Waits for both textures: the depth can arrive before the first image."""
        if not self.allocated or not self.shader_program or not depth.allocated or not image.allocated:
            return

        glUseProgram(self.shader_program)
        depth.bind(0)
        image.bind(1)
        glUniform1f(self.get_uniform_loc("nearDepth"), near)
        glUniform1f(self.get_uniform_loc("farDepth"), far)
        glUniform1i(self.get_uniform_loc("mode"), mode)
        glUniform1i(self.get_uniform_loc("map"), colormap)
        draw_quad()
