# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType
from modules.gl import Fbo, Texture, Image, Blit, clear_color, draw_box_string, text_init
from modules.pose.features.Angles import ANGLE_NUM_LANDMARKS, ANGLE_LANDMARK_NAMES
from modules.pose.pd_stream.PDStream import PDStreamData
from modules.render.layers.LayerBase import LayerBase, DataCache, Rect
from modules.render.shaders import PoseAngleWindow as shader
from .Colors import POSE_COLOR_LEFT, POSE_COLOR_RIGHT, POSE_COLOR_CENTER

from modules.utils.HotReloadMethods import HotReloadMethods


class PDLayer(LayerBase):


    def __init__(self, cam_id: int, data: DataHub) -> None:
        self._cam_id: int = cam_id
        self._data: DataHub = data
        self._fbo: Fbo = Fbo()
        self._label_fbo: Fbo = Fbo()
        self._image: Image = Image()
        self._data_cache: DataCache[PDStreamData] = DataCache[PDStreamData]()

        self.draw_labels: bool = True

        self._shader: shader = shader()
        hot_reload = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self._fbo.texture

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._label_fbo.allocate(width, height, internal_format)
        PDLayer.render_labels(self._label_fbo)
        self._shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._label_fbo.deallocate()
        self._shader.deallocate()

    def draw(self) -> None:
        if self._fbo.allocated:
            Blit.use(self._fbo.texture)
            if self.draw_labels:
                Blit.use(self._label_fbo.texture)

    def update(self) -> None:
        pd_stream: PDStreamData | None = self._data.get_item(DataHubType.pd_stream, self._cam_id)
        self._data_cache.update(pd_stream)

        # We dont reset the FBO on lost PD stream, to keep the last valid image visible
        # if self._data_cache.lost:
        #     self._fbo.clear()

        if self._data_cache.idle or pd_stream is None:
            return

        stream_image: np.ndarray = self._shader.pose_stream_to_image(pd_stream)
        self._image.set_image(stream_image)
        self._image.update()

        self._fbo.begin()
        clear_color()
        self._shader.use(self._image, self._image.width, self._image.height, line_width=1.5 / self._fbo.height)
        self._fbo.end()

    @staticmethod
    def render_labels(fbo: Fbo) -> None:
        text_init()

        rect = Rect(0, 0, fbo.width, fbo.height)

        fbo.begin()
        clear_color()

        angle_num: int = ANGLE_NUM_LANDMARKS
        step: float = rect.height / angle_num
        # yellow and light blue
        colors: list[tuple[float, float, float, float]] = [(*POSE_COLOR_LEFT, 1.0), (*POSE_COLOR_RIGHT, 1.0)]

        for i in range(angle_num):
            string: str = ANGLE_LANDMARK_NAMES[i]
            x: int = int(rect.x + 10)
            y: int = int(rect.y + rect.height - (rect.height - (i + 0.5) * step) - 9)
            clr: int = i % 2

            draw_box_string(x, y, string, colors[clr], (0.0, 0.0, 0.0, 0.3)) # type: ignore

        fbo.end()