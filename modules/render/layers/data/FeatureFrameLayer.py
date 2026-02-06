# Standard library imports
from dataclasses import dataclass
from typing import Tuple

# Third-party imports
from OpenGL.GL import *  # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes
from modules.gl import Fbo, Texture, Blit, clear_color, draw_box_string, text_init
from modules.pose.features import PoseFeatureType
from modules.pose.Frame import Frame, FrameField
from modules.render.layers.LayerBase import LayerBase, DataCache, Rect
from modules.render.shaders import FeatureBand
from .Colors import ANGLES_COLORS, MOVEMENT_COLORS, SIMILARITY_COLORS, BBOX_COLORS

from modules.utils.HotReloadMethods import HotReloadMethods


@dataclass
class FrameLayerConfig:
    """Configuration for FeatureFrameLayer variants."""
    data_type: PoseDataHubTypes
    feature_field: FrameField
    display_range: Tuple[float, float] | None  # None means dynamic from feature.range()
    colors: list[tuple[float, float, float, float]]
    alpha: float
    render_labels: bool = True
    use_scores: bool = False


class FeatureFrameLayer(LayerBase):
    """Visualizes a single pose feature as horizontal lines per joint.

    Displays feature values as horizontal lines with color cycling through
    a configurable color list. Transparent background.
    """

    def __init__(self, track_id: int, data_hub: DataHub, config: FrameLayerConfig,
                 line_thickness: float = 1.0, line_smooth: float = 1.0) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._config: FrameLayerConfig = config

        self._fbo: Fbo = Fbo()
        self._label_fbo: Fbo = Fbo()
        self._data_cache: DataCache[Frame] = DataCache[Frame]()
        self._labels: list[str] = []

        self.line_thickness: float = line_thickness
        self.line_smooth: float = line_smooth
        self.draw_labels: bool = True

        self._shader: FeatureBand = FeatureBand()

        text_init()

        self._hot_reloader = HotReloadMethods(self.__class__, True, True)

    @property
    def texture(self) -> Texture:
        return self._fbo.texture

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._fbo.allocate(width, height, internal_format)
        self._label_fbo.allocate(width, height, internal_format)
        self._shader.allocate()

    def deallocate(self) -> None:
        self._fbo.deallocate()
        self._label_fbo.deallocate()
        self._shader.deallocate()

    def draw(self) -> None:
        if self._fbo.allocated:
            Blit.use(self._fbo.texture)
            if self.draw_labels and self._config.render_labels:
                Blit.use(self._label_fbo.texture)

    def update(self) -> None:
        """Update visualization from DataHub Frame."""
        pose: Frame | None = self._data_hub.get_item(DataHubType(self._config.data_type), self._track_id)
        self._data_cache.update(pose)

        if self._data_cache.lost:
            self._fbo.clear()

        if self._data_cache.idle or pose is None:
            return

        # Extract feature from frame
        feature = pose.get_feature(self._config.feature_field)
        if not isinstance(feature, PoseFeatureType):
            raise ValueError(f"FeatureFrameLayer expected PoseFeatureType, got {type(feature)}")

        # Apply alpha to colors
        colors_with_alpha = [
            (r, g, b, a * self._config.alpha)
            for r, g, b, a in self._config.colors
        ]

        # Override display range if configured
        if self._config.display_range is not None:
            # Temporarily override feature range for rendering
            original_range = feature.range()
            feature._range = self._config.display_range  # type: ignore

        line_thickness = 1.0 / self._fbo.height * self.line_thickness
        line_smooth = 1.0 / self._fbo.height * self.line_smooth

        self._fbo.begin()
        clear_color()
        self._shader.use(feature, colors_with_alpha, line_thickness, line_smooth, self._config.use_scores)
        self._fbo.end()

        # Restore original range
        if self._config.display_range is not None:
            feature._range = original_range  # type: ignore

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
        text_init()

        rect = Rect(0, 0, fbo.width, fbo.height)

        fbo.begin()
        clear_color()

        num_labels: int = len(labels)
        if num_labels == 0:
            fbo.end()
            return

        step: float = rect.width / num_labels

        # Alternate colors for readability
        colors: list[tuple[float, float, float, float]] = ANGLES_COLORS

        for i in range(num_labels):
            string: str = labels[i]
            x: int = int(rect.x + (i + 0.1) * step)
            y: int = int(rect.y + rect.height * 0.5 - 9)
            clr: int = i % 2

            draw_box_string(x, y, string, colors[clr], (0.0, 0.0, 0.0, 0.3))  # type: ignore

        fbo.end()


# Convenience classes for common configurations

class AngleFrameLayer(FeatureFrameLayer):
    """Angle frame layer."""

    def __init__(self, track_id: int, data_hub: DataHub, data_type: PoseDataHubTypes,
                 line_thickness: float = 1.0, line_smooth: float = 1.0,
                 display_range: Tuple[float, float] | None = None) -> None:
        config = FrameLayerConfig(
            data_type=data_type,
            feature_field=FrameField.angles,
            display_range=display_range,
            colors=ANGLES_COLORS,
            alpha=1.0,
            render_labels=True
        )
        super().__init__(track_id, data_hub, config, line_thickness, line_smooth)


class AngleVelFrameLayer(FeatureFrameLayer):
    """Angle velocity frame layer."""

    def __init__(self, track_id: int, data_hub: DataHub, data_type: PoseDataHubTypes,
                 line_thickness: float = 1.0, line_smooth: float = 1.0,
                 display_range: Tuple[float, float] | None = None) -> None:
        import numpy as np
        config = FrameLayerConfig(
            data_type=data_type,
            feature_field=FrameField.angle_vel,
            display_range=display_range if display_range is not None else (-np.pi, np.pi),
            colors=ANGLES_COLORS,
            alpha=1.0,
            render_labels=True
        )
        super().__init__(track_id, data_hub, config, line_thickness, line_smooth)


class AngleMotionFrameLayer(FeatureFrameLayer):
    """Angle motion frame layer."""

    def __init__(self, track_id: int, data_hub: DataHub, data_type: PoseDataHubTypes,
                 line_thickness: float = 1.0, line_smooth: float = 1.0,
                 display_range: Tuple[float, float] | None = (0.0, 5.0)) -> None:
        config = FrameLayerConfig(
            data_type=data_type,
            feature_field=FrameField.angle_motion,
            display_range=display_range,
            colors=MOVEMENT_COLORS,
            alpha=1.0,
            render_labels=True
        )
        super().__init__(track_id, data_hub, config, line_thickness, line_smooth)


class SimilarityFrameLayer(FeatureFrameLayer):
    """Similarity frame layer."""

    def __init__(self, track_id: int, data_hub: DataHub, data_type: PoseDataHubTypes,
                 line_thickness: float = 1.0, line_smooth: float = 1.0,
                 display_range: Tuple[float, float] | None = (0.0, 1.0)) -> None:
        config = FrameLayerConfig(
            data_type=data_type,
            feature_field=FrameField.similarity,
            display_range=display_range,
            colors=SIMILARITY_COLORS,
            alpha=1.0,
            render_labels=False
        )
        super().__init__(track_id, data_hub, config, line_thickness, line_smooth)


class BBoxFrameLayer(FeatureFrameLayer):
    """Bounding box frame layer."""

    def __init__(self, track_id: int, data_hub: DataHub, data_type: PoseDataHubTypes,
                 line_thickness: float = 1.0, line_smooth: float = 1.0,
                 display_range: Tuple[float, float] | None = None) -> None:
        config = FrameLayerConfig(
            data_type=data_type,
            feature_field=FrameField.bbox,
            display_range=display_range,
            colors=BBOX_COLORS,
            alpha=1.0,
            render_labels=True
        )
        super().__init__(track_id, data_hub, config, line_thickness, line_smooth)
