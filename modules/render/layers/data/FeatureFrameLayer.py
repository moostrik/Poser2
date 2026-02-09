# Standard library imports
from dataclasses import dataclass
from typing import Tuple

# Third-party imports
from OpenGL.GL import *  # type: ignore

# Local application imports
from modules.DataHub import DataHub, Stage
from modules.gl import Fbo, Texture, Blit, clear_color, Text
from modules.pose.features import PoseFeatureType
from modules.pose.Frame import Frame, FrameField
from modules.render.layers.LayerBase import LayerBase, DataCache, Rect
from modules.render.shaders import FeatureShader
from modules.render.layers.data.DataLayerConfig import FEATURE_COLORS, DEFAULT_COLORS, DataLayerConfig

from modules.utils.HotReloadMethods import HotReloadMethods


class FeatureFrameLayer(LayerBase):
    """Visualizes a single pose feature as horizontal lines per joint.

    Displays feature values as horizontal lines with color cycling through
    a configurable color list. Transparent background.
    """

    def __init__(self, track_id: int, data_hub: DataHub, config: DataLayerConfig) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._config: DataLayerConfig = config

        self._fbo: Fbo = Fbo()
        self._label_fbo: Fbo = Fbo()
        self._data_cache: DataCache[Frame] = DataCache[Frame]()
        self._labels: list[str] = []

        self.draw_labels: bool = False

        self._shader: FeatureShader = FeatureShader()
        self._text_renderer: Text = Text()

        # Watch active to clear cache on deactivate
        self._config.watch(self._on_active_change, 'active')

        self._hot_reloader = HotReloadMethods(self.__class__, True, True)

    def _on_active_change(self, active: bool) -> None:
        if not active:
            self.clear()

    @property
    def texture(self) -> Texture:
        return self._fbo.texture

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._label_fbo.allocate(width, height, internal_format)
        self._shader.allocate()
        self._text_renderer.allocate("files/RobotoMono-Regular.ttf", font_size=14)

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._label_fbo.deallocate()
        self._shader.deallocate()
        self._text_renderer.deallocate()

    def clear(self) -> None:
        """Clear cached data."""
        self._data_cache = DataCache()
        self._labels = []

    def draw(self) -> None:
        if not self._config.active:
            return
        if self._fbo.allocated:
            Blit.use(self._fbo.texture)
            if self.draw_labels and self._config.render_labels:
                Blit.use(self._label_fbo.texture)

    def update(self) -> None:
        """Update visualization from DataHub Frame."""
        if not self._config.active:
            return
        pose: Frame | None = self._data_hub.get_pose(self._config.stage, self._track_id)
        self._data_cache.update(pose)

        if self._data_cache.lost:
            self._fbo.clear()

        if self._data_cache.idle or pose is None:
            return

        # Extract feature from frame
        feature = pose.get_feature(self._config.feature_field)
        if not isinstance(feature, PoseFeatureType):
            raise ValueError(f"FeatureFrameLayer expected PoseFeatureType, got {type(feature)}")

        # Use config colors or fallback to FEATURE_COLORS
        colors = self._config.colors or FEATURE_COLORS.get(self._config.feature_field, DEFAULT_COLORS)

        line_width = 1.0 / self._fbo.height * self._config.line_width
        line_smooth = 1.0 / self._fbo.height * self._config.line_smooth
        display_range = feature.display_range()

        self._fbo.begin()
        clear_color()
        self._shader.use(feature, colors, line_width, line_smooth, self._config.use_scores, display_range)
        self._fbo.end()

        # Render labels if changed
        if self._config.render_labels:
            joint_enum_type = feature.__class__.enum()
            num_joints: int = len(feature)
            labels: list[str] = [joint_enum_type(i).name for i in range(num_joints)]
            if labels != self._labels:
                self._render_labels_static(self._label_fbo, labels)
                self._labels = labels

    def _render_labels_static(self, fbo: Fbo, labels: list[str]) -> None:
        """Render feature labels overlay."""
        rect = Rect(0, 0, fbo.width, fbo.height)

        fbo.begin()
        clear_color()

        num_labels: int = len(labels)
        if num_labels == 0:
            fbo.end()
            return

        step: float = rect.width / num_labels

        # Use config colors or fallback to FEATURE_COLORS
        colors = self._config.colors or FEATURE_COLORS.get(self._config.feature_field, DEFAULT_COLORS)

        for i in range(num_labels):
            string: str = labels[i]
            x: int = int(rect.x + (i + 0.1) * step)
            y: int = int(rect.y + rect.height * 0.5 - 7)
            clr: int = i % len(colors)

            self._text_renderer.draw_box_text(
                x, y, string, colors[clr], (0.0, 0.0, 0.0, 0.66),
                screen_width=fbo.width, screen_height=fbo.height
            )

        fbo.end()

