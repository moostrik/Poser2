# Standard library imports

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Image import Image
from modules.gl.Text import draw_box_string, text_init

from modules.WS.WSOutput import WSOutput
from modules.pose.similarity.SimilarityStream import SimilarityStreamData
from modules.DataHub import DataHub
from modules.gl.LayerBase import LayerBase, Rect

from modules.gl.shaders.WS_Lines import WS_Lines

class WSLinesLayer(LayerBase):
    lines_shader = WS_Lines()

    def __init__(self, data: DataHub) -> None:
        self.data: DataHub = data
        self.data_consumer_key: str = data.get_unique_consumer_key()
        self.fbo_lines: Fbo = Fbo()
        self.image: Image = Image()
        text_init()

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo_lines.allocate(width, height, internal_format)
        if not WSLinesLayer.lines_shader.allocated:
            WSLinesLayer.lines_shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self.fbo_lines.deallocate()
        self.image.deallocate()
        if WSLinesLayer.lines_shader.allocated:
            WSLinesLayer.lines_shader.deallocate()

    def draw(self, rect: Rect) -> None:
        x, y, width, height = rect.x, rect.y, rect.width, rect.height
        self.fbo_lines.draw(x, y, width, height)

    def update(self) -> None:
        light_image: WSOutput | None = self.data.get_light_image(True, self.data_consumer_key)
        if light_image is None:
            return

        self.image.set_image(light_image.infos_img)
        self.image.update()

        self.setView(self.fbo_lines.width, self.fbo_lines.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.fbo_lines.begin()
        self.lines_shader.use(self.fbo_lines.fbo_id, self.image.tex_id)
        self.fbo_lines.end()