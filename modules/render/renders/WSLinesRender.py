# Standard library imports

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Image import Image
from modules.gl.Text import draw_box_string, text_init

from modules.WS.WSDefinitions import WSOutput
from modules.correlation.PairCorrelationStream import PairCorrelationStreamData
from modules.render.DataManager import DataManager
from modules.render.renders.BaseRender import BaseRender, Rect

from modules.gl.shaders.WS_Lines import WS_Lines

class WSLinesRender(BaseRender):
    lines_shader = WS_Lines()

    def __init__(self, data: DataManager) -> None:
        self.data: DataManager = data
        self.fbo_lines: Fbo = Fbo()
        self.image: Image = Image()
        text_init()

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo_lines.allocate(width, height, internal_format)
        if not WSLinesRender.lines_shader.allocated:
            WSLinesRender.lines_shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self.fbo_lines.deallocate()
        self.image.deallocate()
        if WSLinesRender.lines_shader.allocated:
            WSLinesRender.lines_shader.deallocate()

    def draw(self, rect: Rect) -> None:
        x, y, width, height = rect.x, rect.y, rect.width, rect.height
        self.fbo_lines.draw(x, y, width, height)

    def update(self) -> None:
        light_image: WSOutput | None = self.data.get_light_image(True, self.key())
        if light_image is None:
            return

        self.image.set_image(light_image.infos_img)
        self.image.update()

        self.setView(self.fbo_lines.width, self.fbo_lines.height)
        self.fbo_lines.begin()
        self.lines_shader.use(self.fbo_lines.fbo_id, self.image.tex_id)
        self.fbo_lines.end()