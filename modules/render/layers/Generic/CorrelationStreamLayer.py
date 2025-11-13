# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Image import Image
from modules.gl.Text import draw_box_string, text_init

from modules.pose.similarity.SimilarityStream import SimilarityStream, SimilarityStreamData, SimilarityBatch , AggregationMethod
from modules.gl.LayerBase import LayerBase, Rect
from modules.DataHub import DataHub

from modules.utils.HotReloadMethods import HotReloadMethods

# Shaders
from modules.gl.shaders.StreamCorrelation import StreamCorrelation


class CorrelationStreamLayer(LayerBase):
    r_stream_shader = StreamCorrelation()

    def __init__(self, data: DataHub, num_streams: int, capacity: int, use_motion: bool = False) -> None:
        self.data: DataHub = data
        self.data_consumer_key: str = data.get_unique_consumer_key()
        self.fbo: Fbo = Fbo()
        self.image: Image = Image()
        self.num_streams: int = num_streams
        self.correlation_stream: SimilarityStream = SimilarityStream(capacity)
        self.use_motion: bool = use_motion

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
        """Update and render correlation streams.
        Pipeline:
        1. Fetch new correlation batch from data hub
        2. Update circular buffer with new similarities
        3. Extract top N pairs by average similarity
        4. Convert to GPU texture (num_streams × capacity × RGB)
        5. Render with shader and overlay pair ID labels

        The FBO is always cleared, even if no data is available.
        """
        # reallocate shader if needed if hot-reloaded
        if not CorrelationStreamLayer.r_stream_shader.allocated:
            CorrelationStreamLayer.r_stream_shader.allocate(monitor_file=True)

        if self.use_motion:
            correlation_batch: SimilarityBatch  | None = self.data.get_motion_correlation(True, self.data_consumer_key)
        else:
            correlation_batch: SimilarityBatch  | None = self.data.get_pose_correlation(True, self.data_consumer_key)

        if correlation_batch is None:
            return

        self.correlation_stream.update(correlation_batch, AggregationMethod.GEOMETRIC_MEAN)
        stream_data: SimilarityStreamData = self.correlation_stream.get_stream_data()

        LayerBase.setView(self.fbo.width, self.fbo.height)
        self.fbo.begin()
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        self.fbo.end()

        if stream_data.is_empty:
            return

        pairs: list[tuple[int, int]] = stream_data.get_top_pairs(self.num_streams)
        if pairs == []:
            print("CorrelationStreamLayer.update: No valid pairs found.")
            return
        pair_arrays: list[np.ndarray] = [stream_data.get_similarities(pair_id) for pair_id in pairs]
        if pair_arrays == []:
            print("CorrelationStreamLayer.update: No valid similarity arrays found.")
            return
        image_np: np.ndarray = StreamCorrelation.r_stream_to_image(pair_arrays, self.num_streams)
        self.image.set_image(image_np)
        self.image.update()

        LayerBase.setView(self.fbo.width, self.fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.r_stream_shader.use(self.fbo.fbo_id, self.image.tex_id, self.image.width, self.image.height, 1.5 / self.fbo.height)

        step: float = self.fbo.height / self.num_streams

        self.fbo.begin()
        glColor4f(1.0, 0.5, 0.5, 1.0)
        for i, pair in enumerate(pairs):
            string: str = f'{pair[0]} | {pair[1]}'
            x: int = self.fbo.width - 100
            y: int = int((i + 0.5) * step) + 12
            draw_box_string(x, y, string, big=True)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        self.fbo.end()
