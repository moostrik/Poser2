# Standard library imports
from abc import abstractmethod

# Third-party imports
import numpy as np
from OpenGL.GL import *  # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType
from modules.gl import Fbo, Texture, Blit, Image, clear_color
from modules.pose.nodes import FeatureWindow
from modules.render.layers.LayerBase import LayerBase, DataCache
from modules.render.shaders.window import WindowShaderBase


class WindowLayerBase(LayerBase):
    """Base class for visualizing FeatureWindow data as horizontal time series.

    Converts per-track FeatureWindow (numpy arrays) to GL_RG32F texture where:
    - R channel = feature value
    - G channel = validity mask (0.0 or 1.0)

    Displays feature values as horizontal lines (one per feature element),
    with time flowing left-to-right.

    Subclasses must implement:
    - get_shader() - return WindowShaderBase subclass instance
    - get_feature_names() - return list of feature element names
    - get_display_range() - return (min, max) value range for visualization
    - render_labels_static(fbo) - render feature name labels overlay
    """

    def __init__(self, track_id: int, data_hub: DataHub, data_type: DataHubType, line_width: float) -> None:
        """Initialize feature window layer.

        Args:
            track_id: Track ID to visualize
            data_hub: DataHub instance for data access
            data_type: DataHubType enum value for window data (e.g., angle_vel_window)
            line_width: Width of the lines to draw
        """

        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._data_type: DataHubType = data_type

        self._fbo: Fbo = Fbo()
        self._label_fbo: Fbo = Fbo()
        self._image: Image = Image(channel_order='RGB')
        self._data_cache: DataCache[FeatureWindow] = DataCache()

        self.draw_labels: bool = True
        self.line_width: float = 3.0

        self._shader: WindowShaderBase = self.get_shader()

    @abstractmethod
    def get_shader(self) -> WindowShaderBase:
        """Return shader instance for rendering this feature type."""
        pass

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Return list of feature element names for labels."""
        pass

    @abstractmethod
    def get_display_range(self) -> tuple[float, float]:
        """Return (min, max) value range for visualization."""
        pass

    @abstractmethod
    def render_labels_static(self, fbo: Fbo) -> None:
        """Render feature name labels overlay to FBO."""
        pass

    @property
    def texture(self) -> Texture:
        return self._fbo.texture

    def allocate(self, width: int | None = None, height: int | None = None, internal_format: int | None = None) -> None:
        if width is None or height is None or internal_format is None:
            return
        # self._image.allocate(width, height, GL_RG32F, min_filter=GL_NEAREST, mag_filter=GL_NEAREST)
        self._fbo.allocate(width, height, internal_format)
        self._label_fbo.allocate(width, height, internal_format)
        self.render_labels_static(self._label_fbo)
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
        # Get window from DataHub
        window: FeatureWindow | None = self._data_hub.get_item(self._data_type, self._track_id)
        self._data_cache.update(window)

        if self._data_cache.idle or window is None:
            return

        # Convert numpy arrays to image format: (feature_len, time, 2)
        # Transpose to get horizontal strips: (time, feature_len) -> (feature_len, time)
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

        # Render using feature-specific shader
        self._fbo.begin()
        clear_color()
        num_samples = stream_image.shape[1]  # time (width)
        num_streams = stream_image.shape[0]  # feature_len (height)
        output_aspect = self._fbo.width / self._fbo.height
        display_range = self.get_display_range()[1]  # max absolute value
        self._shader.use(
            self._image.texture,
            num_samples,
            num_streams,
            line_width=self.line_width / self._fbo.height,
            output_aspect_ratio=output_aspect,
            display_range=display_range
        )
        self._fbo.end()
