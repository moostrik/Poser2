# Standard library imports
import numpy as np

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Fbo import Fbo
from modules.gl.Image import Image
from modules.gl.Text import draw_box_string, text_init

from modules.pose.similarity.features.SimilarityStream import SimilarityStream, SimilarityStreamData, SimilarityBatch , AggregationMethod
from modules.gl.LayerBase import LayerBase, Rect
from modules.DataHub import DataHub, DataHubType, SimilarityDataHubType

from modules.utils.HotReloadMethods import HotReloadMethods

# Shaders
from modules.gl.shaders.StreamCorrelation import StreamCorrelation


class SimilarityLineLayer(LayerBase):
    r_stream_shader = StreamCorrelation()

    def __init__(self, num_streams: int, capacity: int, data: DataHub, data_type: SimilarityDataHubType,
                 aggregation_method: AggregationMethod = AggregationMethod.HARMONIC_MEAN, exponent: float = 2.0) -> None:
        self._num_streams: int = num_streams
        self._data: DataHub = data
        self._fbo: Fbo = Fbo()
        self._image: Image = Image()
        self._correlation_stream: SimilarityStream = SimilarityStream(capacity)
        self._p_batch: SimilarityBatch | None = None

        self.data_type: SimilarityDataHubType = data_type
        self.exponent: float = exponent
        self.aggregation_method: AggregationMethod = aggregation_method

        text_init()

        # hot reloader
        self.hot_reloader = HotReloadMethods(self.__class__, True, True)

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        if not SimilarityLineLayer.r_stream_shader.allocated:
            SimilarityLineLayer.r_stream_shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._image.deallocate()
        if SimilarityLineLayer.r_stream_shader.allocated:
            SimilarityLineLayer.r_stream_shader.deallocate()

    def draw(self, rect: Rect) -> None:
        self._fbo.draw(rect.x, rect.y, rect.width, rect.height)

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
        if not SimilarityLineLayer.r_stream_shader.allocated:
            SimilarityLineLayer.r_stream_shader.allocate()

        batch: SimilarityBatch  | None = self._data.get_item(DataHubType(self.data_type))

        # print("yes", batch)
        if batch is self._p_batch:
            # No new data, skip update
            return
        self._p_batch = batch

        LayerBase.setView(self._fbo.width, self._fbo.height)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self._fbo.clear(0.0, 0.0, 0.0, 0.0)

        if batch is None:
            return

        self._correlation_stream.update(batch, self.aggregation_method, 0.0, self.exponent)
        stream_data: SimilarityStreamData = self._correlation_stream.get_stream_data()
        if stream_data.is_empty:
            return

        pairs: list[tuple[int, int]] = stream_data.get_top_pairs(self._num_streams)
        if pairs == []:
            return

        pair_arrays: list[np.ndarray] = [stream_data.get_similarities(pair_id) for pair_id in pairs]
        image_np: np.ndarray = StreamCorrelation.r_stream_to_image(pair_arrays, self._num_streams)
        self._image.set_image(image_np)
        self._image.update()

        self.r_stream_shader.use(self._fbo.fbo_id, self._image.tex_id, self._image.width, self._image.height, 1.5 / self._fbo.height)

        step: float = self._fbo.height / self._num_streams

        self._fbo.begin()
        glColor4f(1.0, 0.5, 0.5, 1.0)
        for i, pair in enumerate(pairs):
            string: str = f'{pair[0]} | {pair[1]}'
            x: int = self._fbo.width - 100
            y: int = int((i + 0.5) * step) + 12
            draw_box_string(x, y, string, big=True)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        self._fbo.end()
