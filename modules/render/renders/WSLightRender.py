# Standard library imports

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Image import Image
from modules.gl.Text import draw_box_string, text_init

from modules.av.Definitions import AvOutput
from modules.correlation.PairCorrelationStream import PairCorrelationStreamData
from modules.render.DataManager import DataManager
from modules.render.renders.BaseRender import BaseRender, Rect

from modules.gl.shaders.WS_Angles import WS_Angles
from modules.gl.shaders.WS_Lines import WS_Lines

class WSLightRender(BaseRender):
    angles_shader = WS_Angles()

    def __init__(self, data: DataManager) -> None:
        self.data: DataManager = data
        self.fbo_angles: Fbo = Fbo()
        self.image: Image = Image()
        text_init()

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo_angles.allocate(width, height, internal_format)
        if not WSLightRender.angles_shader.allocated:
            WSLightRender.angles_shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self.fbo_angles.deallocate()
        self.image.deallocate()
        if WSLightRender.angles_shader.allocated:
            WSLightRender.angles_shader.deallocate()

    def draw(self, rect: Rect) -> None:
        x, y, width, height = rect.x, rect.y, rect.width, rect.height
        self.fbo_angles.draw(x, y, width, height)

    def update(self) -> None:
        light_image: AvOutput | None = self.data.get_light_image(True, self.key())
        if light_image is None:
            return

        self.image.set_image(light_image.img)
        self.image.update()

        self.setView(self.fbo_angles.width, self.fbo_angles.height)
        self.fbo_angles.begin()
        self.angles_shader.use(self.fbo_angles.fbo_id, self.image.tex_id, self.fbo_angles.width)
        self.fbo_angles.end()