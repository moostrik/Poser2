from OpenGL.GL import * # type: ignore

from modules.gl import Fbo, Texture, Image
from modules.board import HasCompositionDebug
from modules.render.layers.LayerBase import LayerBase

from apps.white_space.render.shaders.WS_Lines import WS_Lines


class CompositionDebugLayer(LayerBase):

    def __init__(self, board: HasCompositionDebug) -> None:
        self.board: HasCompositionDebug = board
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
        debug = self.board.get_composition_debug()
        if debug is None:
            return

        self.image.set_image(debug.debug_img)
        self.image.update()

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.fbo_lines.begin()
        self._shader.use(self.fbo_lines, self.image.texture)
        self.fbo_lines.end()
