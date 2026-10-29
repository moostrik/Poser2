from OpenGL.GL import * # type: ignore

from modules.gl import Fbo, Texture, Image
from modules.render import layers

from apps.white_space.render.shaders.light_simulation import LightSimulation

from ...board import HasLightFrame


class LightSimulationLayer(layers.LayerBase):

    def __init__(self, board: HasLightFrame) -> None:
        self.board: HasLightFrame = board
        self.fbo_angles: Fbo = Fbo()
        self.image: Image = Image()
        self._shader: LightSimulation = LightSimulation()

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
        frame = self.board.get_light_frame()
        if frame is None:
            return

        self.image.set_image(frame.light_img)
        self.image.update()

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.fbo_angles.begin()
        self._shader.use(self.fbo_angles, self.image.texture, self.fbo_angles.width)
        self.fbo_angles.end()
