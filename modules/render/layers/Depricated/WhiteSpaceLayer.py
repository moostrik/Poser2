# Standard library imports

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Image import Image
from modules.gl.Text import draw_box_string, text_init

from modules.WS.WSOutput import WSOutput
from modules.pose.similarity.features.SimilarityStream import SimilarityStream
from modules.DataHub import DataHub
from modules.gl.LayerBase import LayerBase, Rect

from modules.gl.shaders.WS_Angles import WS_Angles
from modules.gl.shaders.WS_Lines import WS_Lines

class WhiteSpaceRender(LayerBase):
    angles_shader = WS_Angles()
    lines_shader = WS_Lines()

    def __init__(self, data: DataHub) -> None:
        self.data: DataHub = data
        self.data_consumer_key: str = data.get_unique_consumer_key()
        self.fbo_lines: Fbo = Fbo()
        self.fbo_angles: Fbo = Fbo()
        self.image: Image = Image()
        text_init()

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo_lines.allocate(width, height, internal_format)
        self.fbo_angles.allocate(width, height, internal_format)
        if not WhiteSpaceRender.lines_shader.allocated:
            WhiteSpaceRender.lines_shader.allocate(monitor_file=True)
        if not WhiteSpaceRender.angles_shader.allocated:
            WhiteSpaceRender.angles_shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self.fbo_lines.deallocate()
        self.fbo_angles.deallocate()
        self.image.deallocate()
        if WhiteSpaceRender.lines_shader.allocated:
            WhiteSpaceRender.lines_shader.deallocate()
        if WhiteSpaceRender.angles_shader.allocated:
            WhiteSpaceRender.angles_shader.deallocate()

    def draw(self, rect: Rect) -> None:
        x, y, width, height = rect.x, rect.y, rect.width, rect.height
        half_height: float = height / 2
        self.fbo_lines.draw(x, y, width, half_height)
        self.fbo_angles.draw(x, y+half_height, width, half_height)

    def update(self) -> None:
        light_image: WSOutput | None = self.data.get_light_image(True, self.data_consumer_key)
        if light_image is None:
            return

        self.image.set_image(light_image.light_img)
        self.image.update()

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.setView(self.fbo_lines.width, self.fbo_lines.height)
        self.fbo_lines.begin()
        self.lines_shader.use(self.fbo_lines.fbo_id, self.image.tex_id)
        self.fbo_lines.end()

        self.setView(self.fbo_angles.width, self.fbo_angles.height)
        self.fbo_angles.begin()
        self.angles_shader.use(self.fbo_angles.fbo_id, self.image.tex_id, self.fbo_angles.width)
        self.fbo_angles.end()