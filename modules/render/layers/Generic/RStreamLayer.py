# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Image import Image
from modules.gl.Text import draw_box_string, text_init

from modules.correlation.PairCorrelationStream import PairCorrelationStreamData
from modules.render.DataManager import DataManager
from modules.gl.LayerBase import LayerBase, Rect

from modules.utils.HotReloadMethods import HotReloadMethods

# Shaders
from modules.gl.shaders.WS_RStream import WS_RStream

class RStreamLayer(LayerBase):
    r_stream_shader = WS_RStream()

    def __init__(self, data: DataManager, num_streams: int) -> None:
        self.data: DataManager = data
        self.data_consumer_key: str = data.get_unique_consumer_key()
        self.fbo: Fbo = Fbo()
        self.image: Image = Image()
        self.num_streams: int = num_streams
        text_init()
        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo.allocate(width, height, internal_format)
        if not RStreamLayer.r_stream_shader.allocated:
            RStreamLayer.r_stream_shader.allocate(monitor_file=False)

    def deallocate(self) -> None:
        self.fbo.deallocate()
        self.image.deallocate()
        if RStreamLayer.r_stream_shader.allocated:
            RStreamLayer.r_stream_shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self.fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        correlation_streams: PairCorrelationStreamData | None = self.data.get_correlation_streams(True, self.data_consumer_key)
        if correlation_streams is None:
            return

        pairs: list[tuple[int, int]] = correlation_streams.get_top_pairs(self.num_streams)
        num_pairs: int = len(pairs)

        image_np: np.ndarray = WS_RStream.r_stream_to_image(correlation_streams, self.num_streams)
        self.image.set_image(image_np)
        self.image.update()

        LayerBase.setView(self.fbo.width, self.fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        self.fbo.end()


        self.r_stream_shader.use(self.fbo.fbo_id, self.image.tex_id, self.image.width, self.image.height, 1.5 / self.fbo.height)

        step: float = self.fbo.height / self.num_streams

        LayerBase.setView(self.fbo.width, self.fbo.height)

        self.fbo.begin()
        glColor4f(1.0, 0.5, 0.5, 1.0)
        for i in range(num_pairs):
            pair: tuple[int, int] = pairs[i]
            string: str = f'{pair[0]} | {pair[1]}'
            x: int = self.fbo.width - 100
            y: int = self.fbo.height - (int(self.fbo.height - (i + 0.5) * step) - 12)
            draw_box_string(x, y, string, big=True)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        self.fbo.end()
