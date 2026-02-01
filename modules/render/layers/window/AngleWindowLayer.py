# Standard library imports

# Third-party imports
import numpy as np
from OpenGL.GL import *  # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType
from modules.gl import Fbo, Texture, Blit, Image, clear_color, draw_box_string, text_init
from modules.pose.features.Angles import ANGLE_NUM_LANDMARKS
from modules.pose.nodes import FeatureWindow
from modules.render.layers.LayerBase import LayerBase, DataCache, Rect
from modules.render.shaders.window import WindowShader
from ..data.Colors import POSE_COLOR_LEFT, POSE_COLOR_RIGHT

from modules.utils.HotReloadMethods import HotReloadMethods

class AngleWindowLayer(LayerBase):
    """Visualizes angle window for a single track.

    Displays angle values as horizontal lines (one per angle element),
    with time flowing left-to-right. Sources data from AngleWindowTracker
    via DataHub per-track windows.
    """

    def __init__(self, track_id: int, data_hub: DataHub, line_width: float) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._data_type: DataHubType = DataHubType.angle_window

        self._fbo: Fbo = Fbo()
        self._label_fbo: Fbo = Fbo()
        self._image: Image = Image(channel_order='RGB')
        self._data_cache: DataCache[FeatureWindow] = DataCache()

        self.draw_labels: bool = True
        self.line_width: float = line_width

        self._shader: WindowShader = WindowShader()

        self._hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self._fbo.texture

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        if width is None or height is None or internal_format is None:
            return
        self._fbo.allocate(width, height, internal_format)
        self._label_fbo.allocate(width, height, internal_format)
        # Get feature names from first available window for label rendering
        window: FeatureWindow | None = self._data_hub.get_item(self._data_type, self._track_id)
        if window is not None:
            self._render_labels_static(self._label_fbo, window.feature_names)
        self._shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._label_fbo.deallocate()
        self._image.deallocate()
        self._shader.deallocate()

    def draw(self) -> None:
        if self._fbo.allocated:
            Blit.use(self._fbo.texture)
            if self.draw_labels:
                Blit.use(self._label_fbo.texture)

    def update(self) -> None:
        """Update visualization from DataHub FeatureWindow."""
        window: FeatureWindow | None = self._data_hub.get_item(self._data_type, self._track_id)
        self._data_cache.update(window)

        if self._data_cache.idle or window is None:
            return

        # Convert numpy arrays to image format: (feature_len, time, 2)
        values = window.values.T.astype(np.float32)  # (feature_len, time)
        mask = window.mask.astype(np.float32).T      # (feature_len, time)

        # Stack as 2-channel RG texture
        stream_image = np.stack([values, mask], axis=-1)  # (feature_len, time, 2)

        # Flip vertically for OpenGL convention (oldest at bottom)
        stream_image = np.flip(stream_image, axis=0)

        # Upload to GPU via Image (numpy -> GL_RG32F texture)
        self._image.set_image(stream_image)
        self._image.update()

        self._shader.reload()

        # print("AngleWindowLayer update: rendering to FBO")

        # Render using shader
        self._fbo.begin()
        clear_color()
        num_samples = stream_image.shape[1]  # time (width)
        num_streams = stream_image.shape[0]  # feature_len (height)
        output_aspect = self._fbo.width / self._fbo.height
        self._shader.use(
            self._image.texture,
            num_samples,
            num_streams,
            line_width=self.line_width / self._fbo.height,
            output_aspect_ratio=output_aspect,
            display_range=window.range,
            color_even=(1.0, 0.5, 0.0),  # orange
            color_odd=(0.0, 1.0, 1.0),   # cyan
            alpha=0.75
        )
        self._fbo.end()

    def _render_labels_static(self, fbo: Fbo, feature_names: list[str]) -> None:
        """Render angle labels overlay."""
        text_init()

        rect = Rect(0, 0, fbo.width, fbo.height)

        fbo.begin()
        clear_color()

        angle_num: int = ANGLE_NUM_LANDMARKS
        step: float = rect.height / angle_num
        colors: list[tuple[float, float, float, float]] = [
            (*POSE_COLOR_LEFT, 1.0),
            (*POSE_COLOR_RIGHT, 1.0)
        ]

        for i in range(angle_num):
            string: str = feature_names[i]
            x: int = int(rect.x + 10)
            y: int = int(rect.y + rect.height - (rect.height - (i + 0.5) * step) - 9)
            clr: int = i % 2

            draw_box_string(x, y, string, colors[clr], (0.0, 0.0, 0.0, 0.3))  # type: ignore

        fbo.end()
