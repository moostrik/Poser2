# Standard library imports
import numpy as np
import time

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl.Image import Image, Texture

from modules.DataHub import DataHub, DataHubType
from modules.render.layers.LayerBase import LayerBase, DataCache, Rect
from modules.utils.PerformanceTimer import PerformanceTimer


class ImageSourceLayer(LayerBase):
    def __init__(self, cam_id: int, data: DataHub) -> None:
        self._cam_id: int = cam_id
        self._data: DataHub = data
        self._image: Image = Image('BGR')
        self._data_cache: DataCache[np.ndarray]= DataCache[np.ndarray]()

        # Performance timer
        self._update_timer: PerformanceTimer = PerformanceTimer(
            name="ImageSource GL Upload", sample_count=10000, report_interval=100, color="red", omit_init=10
        )

    @property
    def texture(self) -> Texture:
        return self._image

    def deallocate(self) -> None:
        if self._image.allocated:
            self._image.deallocate()

    def update(self) -> None:
        start = time.perf_counter()

        frame: np.ndarray | None = self._data.get_item(DataHubType.cam_image, self._cam_id)
        self._data_cache.update(frame)

        if self._data_cache.lost:
            self._image.clear()

        if self._data_cache.idle or frame is None:
            return

        start = time.perf_counter()
        self._image.set_image(frame)
        self._image.update()

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._update_timer.add_time(elapsed_ms, False)


