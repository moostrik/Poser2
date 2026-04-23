# Standard library imports

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import Fbo, Texture, Image

from modules.WS.WSOutput import WSOutput
from modules.board import HasWSOutput
from modules.render.layers.LayerBase import LayerBase

from apps.white_space.shaders.WS_Lines import WS_Lines

class WSLinesLayer(LayerBase):

    def __init__(self, board: HasWSOutput) -> None:
        self.board: HasWSOutput = board
        self.fbo_lines: Fbo = Fbo()
        self.image: Image = Image()
        self._shader: WS_Lines = WS_Lines()

    @property
    def texture(self) -> Texture:
        return self.fbo_lines

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo_lines.allocate(width, height, internal_format)
        self._shader.allocate()

    def deallocate(self) -> None:
        self.fbo_lines.deallocate()
        self.image.deallocate()
        self._shader.deallocate()

    def update(self) -> None:
        light_image: WSOutput | None = self.board.get_ws_output()
        if light_image is None:
            return

        self.image.set_image(light_image.infos_img)
        self.image.update()

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.fbo_lines.begin()
        self._shader.use(self.fbo_lines, self.image.texture)
        self.fbo_lines.end()
