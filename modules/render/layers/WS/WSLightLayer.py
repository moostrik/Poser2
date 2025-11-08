# Standard library imports

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Image import Image
from modules.gl.Text import draw_box_string, text_init

from modules.WS.WSOutput import WSOutput
from modules.pose.similarity.SimilarityStream import SimilarityStreamData
from modules.data.CaptureDataHub import CaptureDataHub
from modules.gl.LayerBase import LayerBase, Rect

from modules.gl.shaders.WS_Angles import WS_Angles
from modules.gl.shaders.WS_Lines import WS_Lines

class WSLightLayer(LayerBase):
    angles_shader = WS_Angles()

    def __init__(self, data: CaptureDataHub) -> None:
        self.data: CaptureDataHub = data
        self.data_consumer_key: str = data.get_unique_consumer_key()
        self.fbo_angles: Fbo = Fbo()
        self.image: Image = Image()
        text_init()

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo_angles.allocate(width, height, internal_format)
        if not WSLightLayer.angles_shader.allocated:
            WSLightLayer.angles_shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self.fbo_angles.deallocate()
        self.image.deallocate()
        if WSLightLayer.angles_shader.allocated:
            WSLightLayer.angles_shader.deallocate()

    def draw(self, rect: Rect) -> None:
        x, y, width, height = rect.x, rect.y, rect.width, rect.height
        self.fbo_angles.draw(x, y, width, height)

    def update(self) -> None:
        light_image: WSOutput | None = self.data.get_light_image(True, self.data_consumer_key)
        if light_image is None:
            return

        self.image.set_image(light_image.light_img)
        self.image.update()

        self.setView(self.fbo_angles.width, self.fbo_angles.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.fbo_angles.begin()
        self.angles_shader.use(self.fbo_angles.fbo_id, self.image.tex_id, self.fbo_angles.width)
        self.fbo_angles.end()