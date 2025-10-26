# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Image import Image
from modules.gl.Text import draw_box_string, text_init

from modules.pose.correlation.PairCorrelationStream import PairCorrelationStream, PairCorrelationStreamData, PairCorrelationBatch, SimilarityMetric
from modules.gl.LayerBase import LayerBase, Rect
from modules.CaptureDataHub import CaptureDataHub

from modules.utils.HotReloadMethods import HotReloadMethods

# Shaders
from modules.gl.shaders.StreamCorrelation import StreamCorrelation

class CorrelationStreamLayer(LayerBase):
    r_stream_shader = StreamCorrelation()

    def __init__(self, data: CaptureDataHub, num_streams: int) -> None:
        self.data: CaptureDataHub = data
        self.data_consumer_key: str = data.get_unique_consumer_key()
        self.fbo: Fbo = Fbo()
        self.image: Image = Image()
        self.num_streams: int = num_streams
        self.correlation_stream = PairCorrelationStream(10 * 24)

        text_init()

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self.fbo.allocate(width, height, internal_format)
        if not CorrelationStreamLayer.r_stream_shader.allocated:
            CorrelationStreamLayer.r_stream_shader.allocate(monitor_file=True)

    def deallocate(self) -> None:
        self.fbo.deallocate()
        self.image.deallocate()
        if CorrelationStreamLayer.r_stream_shader.allocated:
            CorrelationStreamLayer.r_stream_shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self.fbo.draw(rect.x, rect.y, rect.width, rect.height)

    def update(self) -> None:
        if not CorrelationStreamLayer.r_stream_shader.allocated:
            CorrelationStreamLayer.r_stream_shader.allocate(monitor_file=True)

        correlation_batch: PairCorrelationBatch | None = self.data.get_pose_correlation(True, self.data_consumer_key)
        if correlation_batch is None:
            return

        self.correlation_stream.update(correlation_batch, SimilarityMetric.GEOMETRIC_MEAN)
        stream_data: PairCorrelationStreamData = self.correlation_stream.get_stream_data()

        pairs_with_data = stream_data.get_top_pairs_with_windows(
            self.num_streams,
            stream_data.capacity
        )

        # Extract just the pairs for rendering
        pairs: list[tuple[int, int]] = [pair_id for pair_id, _ in pairs_with_data]
        num_pairs: int = len(pairs)

        image_np: np.ndarray = StreamCorrelation.r_stream_to_image(stream_data, self.num_streams)
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
