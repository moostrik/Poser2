# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Image import Image
from modules.gl.Text import draw_box_string, text_init

from modules.av.Definitions import AvOutput
from modules.correlation.PairCorrelationStream import PairCorrelationStreamData
from modules.render.DataManager import DataManager
from modules.render.Draw.DrawBase import DrawBase, Rect

from modules.gl.shaders.WS_Angles import WS_Angles
from modules.gl.shaders.WS_Lines import WS_Lines

class DrawWhiteSpace(DrawBase):
    angles_shader = WS_Angles()
    lines_shader = WS_Lines()

    def __init__(self, data: DataManager) -> None:
        self.data: DataManager = data
        self.fbo_lines: Fbo = Fbo()
        self.fbo_angles: Fbo = Fbo()
        self.image: Image = Image()

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo_lines.allocate(width, height, internal_format)
        self.fbo_angles.allocate(width, height, internal_format)
        if not DrawWhiteSpace.lines_shader.allocated:
            DrawWhiteSpace.lines_shader.allocate(monitor_file=False)
        if not DrawWhiteSpace.angles_shader.allocated:
            DrawWhiteSpace.angles_shader.allocate(monitor_file=False)

    def deallocate(self) -> None:
        self.fbo_lines.deallocate()
        self.fbo_angles.deallocate()
        self.image.deallocate()
        if DrawWhiteSpace.lines_shader.allocated:
            DrawWhiteSpace.lines_shader.deallocate()
        if DrawWhiteSpace.angles_shader.allocated:
            DrawWhiteSpace.angles_shader.deallocate()

    def update(self, only_if_dirty: bool) -> None:
        light_image: AvOutput | None = self.data.get_light_image(only_if_dirty)
        if light_image is None:
            return

        self.image.set_image(light_image.img)
        self.image.update()

        self.setView(self.fbo_lines.width, self.fbo_lines.height)
        self.fbo_lines.begin()
        self.lines_shader.use(self.fbo_lines.fbo_id, self.image.tex_id)
        self.fbo_lines.end()

        self.setView(self.fbo_angles.width, self.fbo_angles.height)
        self.fbo_angles.begin()
        self.angles_shader.use(self.fbo_angles.fbo_id, self.image.tex_id, self.fbo_angles.width)
        self.fbo_angles.end()

    def draw(self, rect: Rect) -> None:
        x, y, width, height = rect.x, rect.y, rect.width, rect.height
        half_height: float = height / 2
        self.fbo_lines.draw(x, y, width, half_height)
        self.fbo_angles.draw(x, y+half_height, width, half_height)