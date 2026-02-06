# Standard library imports
from dataclasses import dataclass
from typing import Tuple, Optional

# Third-party imports
import numpy as np
from OpenGL.GL import *  # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType
from modules.gl import Fbo, Texture, Blit, Image, clear_color, draw_box_string, text_init
from modules.pose.nodes import FeatureWindow
from modules.render.layers.LayerBase import LayerBase, DataCache, Rect
from modules.render.shaders import WindowShader
from .Colors import ANGLES_COLORS, MOVEMENT_COLORS, SIMILARITY_COLORS, BBOX_COLORS

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class WindowLayerConfig:
    """Configuration for FeatureWindowLayer variants."""
    data_type: DataHubType
    display_range: Tuple[float, float] | None  # None means dynamic from window.range
    colors: list[tuple[float, float, float, float]]  # Cycle through these RGBA colors
    alpha: float
    render_labels: bool = True


class FeatureWindowLayer(LayerBase):
    """Visualizes feature window data for a single track.

    Displays feature values as horizontal lines (one per feature element),
    with time flowing left-to-right. Sources data from various window trackers
    via DataHub per-track windows.
    """

    def __init__(self, track_id: int, data_hub: DataHub, line_width: float, config: WindowLayerConfig) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._config: WindowLayerConfig = config

        self._fbo: Fbo = Fbo()
        self._label_fbo: Fbo = Fbo()
        self._image: Image = Image()
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
        if self._config.render_labels:
            window: FeatureWindow | None = self._data_hub.get_item(self._config.data_type, self._track_id)
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
            if self.draw_labels and self._config.render_labels:
                Blit.use(self._label_fbo.texture)

    def update(self) -> None:
        """Update visualization from DataHub FeatureWindow."""
        window: FeatureWindow | None = self._data_hub.get_item(self._config.data_type, self._track_id)
        self._data_cache.update(window)

        if self._data_cache.idle or window is None:
            return

        # Convert numpy arrays to image format: (feature_len, time, 2)
        values = window.values.T.astype(np.float32)  # (feature_len, time)
        mask = window.mask.astype(np.float32).T      # (feature_len, time)

        # Stack as 2-channel RG texture
        stream_image = np.stack([values, mask], axis=-1)  # (feature_len, time, 2)

        # Upload to GPU via Image (numpy -> GL_RG32F texture)
        self._image.set_image(stream_image)
        self._image.update()

        self._shader.reload()

        # Determine display range (static from config or dynamic from window)
        display_range = self._config.display_range if self._config.display_range is not None else window.range

        # Render using shader
        self._fbo.begin()
        clear_color()
        num_samples = stream_image.shape[1]  # time (width)
        num_streams = stream_image.shape[0]  # feature_len (height)
        output_aspect = self._fbo.width / self._fbo.height

        # Constrain stream height to 20% of layer width
        max_stream_step = 0.2 * output_aspect
        stream_step = min(1.0 / num_streams, max_stream_step)

        self._shader.use(
            self._image.texture,
            num_samples,
            num_streams,
            stream_step,
            line_width=self.line_width / self._fbo.height,
            output_aspect_ratio=output_aspect,
            display_range=display_range,
            colors=self._config.colors,
            alpha=self._config.alpha
        )
        self._fbo.end()

    def _render_labels_static(self, fbo: Fbo, feature_names: list[str]) -> None:
        """Render feature labels overlay."""
        text_init()

        rect = Rect(0, 0, fbo.width, fbo.height)

        fbo.begin()
        clear_color()

        feature_num: int = len(feature_names)
        step: float = rect.height / feature_num
        colors: list[tuple[float, float, float, float]] = ANGLES_COLORS

        for i in range(feature_num):
            string: str = feature_names[i]
            x: int = int(rect.x + 10)
            y: int = int(rect.y + rect.height - (rect.height - (i + 0.5) * step) - 9)
            clr: int = i % 2

            draw_box_string(x, y, string, colors[clr], (0.0, 0.0, 0.0, 0.3))  # type: ignore

        fbo.end()


# Convenience classes for common configurations

class AngleMtnWindowLayer(FeatureWindowLayer):
    """Angle motion window layer."""

    def __init__(self, track_id: int, data_hub: DataHub, line_width: float,
                 display_range: Tuple[float, float] | None = None) -> None:
        config = WindowLayerConfig(
            data_type=DataHubType.angle_motion_window,
            display_range=display_range,
            colors=MOVEMENT_COLORS,
            alpha=1.0,
            render_labels=True
        )
        super().__init__(track_id, data_hub, line_width, config)


class AngleVelWindowLayer(FeatureWindowLayer):
    """Angle velocity window layer."""

    def __init__(self, track_id: int, data_hub: DataHub, line_width: float,
                 display_range: Tuple[float, float] | None = None) -> None:
        config = WindowLayerConfig(
            data_type=DataHubType.angle_vel_window,
            display_range=display_range if display_range is not None else (-np.pi, np.pi),
            colors=ANGLES_COLORS,
            alpha=1.0,
            render_labels=True
        )
        super().__init__(track_id, data_hub, line_width, config)


class AngleWindowLayer(FeatureWindowLayer):
    """Angle window layer."""

    def __init__(self, track_id: int, data_hub: DataHub, line_width: float,
                 display_range: Tuple[float, float] | None = None) -> None:
        config = WindowLayerConfig(
            data_type=DataHubType.angle_window,
            display_range=display_range,  # Default to dynamic from window.range
            colors=ANGLES_COLORS,
            alpha=1.0,
            render_labels=True
        )
        super().__init__(track_id, data_hub, line_width, config)


class SimilarityWindowLayer(FeatureWindowLayer):
    """Similarity window layer."""

    def __init__(self, track_id: int, data_hub: DataHub, line_width: float,
                 display_range: Tuple[float, float] | None = (0.0, 1.0)) -> None:
        config = WindowLayerConfig(
            data_type=DataHubType.similarity_window,
            display_range=display_range,
            colors=SIMILARITY_COLORS,
            alpha=1.0,
            render_labels=False  # Labels were commented out in original
        )
        super().__init__(track_id, data_hub, line_width, config)


class BBoxWindowLayer(FeatureWindowLayer):
    """Bounding box window layer."""

    def __init__(self, track_id: int, data_hub: DataHub, line_width: float,
                 display_range: Tuple[float, float] | None = None) -> None:
        config = WindowLayerConfig(
            data_type=DataHubType.bbox_window,
            display_range=display_range,  # Dynamic from window.range
            colors=BBOX_COLORS,
            alpha=1.0,
            render_labels=True
        )
        super().__init__(track_id, data_hub, line_width, config)
