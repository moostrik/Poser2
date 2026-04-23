from OpenGL.GL import * # type: ignore

from modules.gl import Fbo, Texture, Image
from modules.board import HasLightOutput
from modules.render.layers.LayerBase import LayerBase

from apps.white_space.shaders.WS_Angles import WS_Angles


class LightLayer(LayerBase):

    def __init__(self, board: HasLightOutput) -> None:
        self.board: HasLightOutput = board
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
        output = self.board.get_light_output()
        if output is None:
            return

        self.image.set_image(output.light_img)
        self.image.update()

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.fbo_angles.begin()
        self._shader.use(self.fbo_angles, self.image.texture, self.fbo_angles.width)
        self.fbo_angles.end()
