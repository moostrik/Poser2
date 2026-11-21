# Standard library imports
from enum import IntEnum
from typing import cast

# Third-party imports
import numpy as np

# Local application imports
from modules.board import HasWindows
from modules.gl import Texture, Image, clear_color
from modules.utils import Rect
from modules.pose import FeatureWindow, BaseScalarFeature
from modules.settings import BaseSettings
from ...layer_base import LayerBase
from .shaders import Window
from .data_layer_settings import DataLayerSettingsProtocol, LayerMode, FEATURE_MAP, AngleOffset, has_pi_range, offset_angles
from .feature_plot import FeaturePlot
from ...color_settings import ColorSettings


class FeatureWindowLayer(LayerBase):
    """Visualizes feature window data for a single track.

    Displays feature values as horizontal lines (one per feature element),
    with time flowing left-to-right. Sources data from various window trackers
    via the board's per-track windows.
    """

    def __init__(self, track_id: int, board: HasWindows, settings: BaseSettings, color_settings: ColorSettings,
                 feature_map: dict[IntEnum, type[BaseScalarFeature]] = FEATURE_MAP) -> None:
        self._track_id: int = track_id
        self._board: HasWindows = board
        self.settings: DataLayerSettingsProtocol = cast(DataLayerSettingsProtocol, settings)
        self._plot: FeaturePlot[FeatureWindow] = FeaturePlot(LayerMode.WINDOW, self.settings, color_settings, feature_map,
                                                             draw_labels=True, font_size=30)
        self._image: Image = Image()

        self._width: int = 0
        self._height: int = 0
        self._stream_step_pixels: float = 0.0

        self._shader: Window = Window()

    @property
    def texture(self) -> Texture:
        return self._plot.fbo

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        if width is None or height is None or internal_format is None:
            return
        self._width = width
        self._height = height
        self._plot.allocate(width, height, internal_format)
        self._shader.allocate()

    def deallocate(self) -> None:
        self._plot.deallocate()
        self._image.deallocate()
        self._shader.deallocate()

    def clear(self) -> None:
        """Clear cached data."""
        self._plot.clear()

    def draw(self) -> None:
        self._plot.draw()

    def _label_place(self, i: int, count: int, rect: Rect) -> tuple[int, int]:
        """Label i of a window: at the left, beside its stream."""
        step: float = self._stream_step_pixels
        return int(rect.x + 10), int(rect.y + rect.height - (rect.height - (i + 0.5) * step) - 7)

    def update(self) -> None:
        """Update visualization from the board's FeatureWindow."""
        if not self._plot.update_active():
            return
        # feature_field maps to feature type for the board's lookup
        feature_type = self._plot.feature_map[self.settings.feature_field]
        window: FeatureWindow | None = self._board.get_window(
            self.settings.stage, self._track_id, feature_type
        )
        cache = self._plot.data_cache
        cache.set(window)

        if cache.idle or window is None:
            return

        # Convert numpy arrays to image format: (feature_len, time, 2)
        values = window.values.T.astype(np.float32)  # (feature_len, time)
        offset = self.settings.angle_offset
        if offset != AngleOffset.NONE and has_pi_range(feature_type):
            values = offset_angles(values, offset)
        mask = window.mask.astype(np.float32).T      # (feature_len, time)

        # Stack as 2-channel RG texture
        stream_image = np.stack([values, mask], axis=-1)  # (feature_len, time, 2)

        # Upload to GPU via Image (numpy -> GL_RG32F texture)
        self._image.set_image(stream_image)
        self._image.update()

        # Use window's display_range
        display_range = window.display_range

        # Render using shader
        fbo = self._plot.fbo
        fbo.begin()
        clear_color()
        num_samples = stream_image.shape[1]  # time (width)
        num_streams = stream_image.shape[0]  # feature_len (height)
        output_aspect = fbo.width / fbo.height

        # Constrain stream height to 20% of layer width
        max_stream_step = 0.2 * output_aspect
        stream_step = min(1.0 / num_streams, max_stream_step)
        self._stream_step_pixels = stream_step * self._height

        self._shader.use(
            self._image,
            num_samples,
            num_streams,
            stream_step,
            line_width=self.settings.line_width / fbo.height,
            output_aspect_ratio=output_aspect,
            display_range=display_range,
            colors=self._plot.colors(),
        )
        fbo.end()

        if self.settings.render_labels:
            self._plot.update_labels(window.feature_names, self._label_place)
