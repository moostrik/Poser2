# Standard library imports

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import Fbo, Texture, Image

from modules.WS.WSOutput import WSOutput
from modules.board import HasWSOutput
from modules.render.layers.LayerBase import LayerBase

from modules.render.shaders.ws.WS_Angles import WS_Angles

class WSLightLayer(LayerBase):

    def __init__(self, board: HasWSOutput) -> None:
        self.board: HasWSOutput = board
        self.fbo_angles: Fbo = Fbo()
        self.image: Image = Image()
        self._shader: WS_Angles = WS_Angles()

    @property
    def texture(self) -> Texture:
        return self.fbo_angles

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo_angles.allocate(width, height, internal_format)
        self._shader.allocate()

    def deallocate(self) -> None:
        self.fbo_angles.deallocate()
        self.image.deallocate()
        self._shader.deallocate()

    def update(self) -> None:
        light_image: WSOutput | None = self.board.get_ws_output()
        if light_image is None:
            return

        self.image.set_image(light_image.light_img)
        self.image.update()

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.fbo_angles.begin()
        self._shader.use(self.fbo_angles, self.image.texture, self.fbo_angles.width)
        self.fbo_angles.end()