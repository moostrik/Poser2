# Standard library imports
from enum import IntEnum
from typing import cast

# Local application imports
from modules.board import HasFrames
from modules.gl import Texture, clear_color
from modules.utils import Rect
from modules.pose import AngleVelocity, Angles, BaseScalarFeature, Frame
from modules.settings import BaseSettings
from ....layer_base import LayerBase
from ..shaders import Feature
from .data_layer_settings import DataLayerSettingsProtocol, LayerMode, FEATURE_MAP, AngleOffset, has_pi_range, offset_angles
from .feature_plot import FeaturePlot
from ....color_settings import ColorSettings


def _label_place(i: int, count: int, rect: Rect) -> tuple[int, int]:
    """Label i of a joint row: in its joint's column, halfway up."""
    step: float = rect.width / count
    return int(rect.x + (i + 0.1) * step), int(rect.y + rect.height * 0.5 - 7)


class FeatureFrameLayer(LayerBase):
    """Visualizes a single pose feature as horizontal lines per joint.

    Displays feature values as horizontal lines with color cycling through
    a configurable color list. Transparent background.
    """

    def __init__(self, track_id: int, board: HasFrames, settings: BaseSettings, color_settings: ColorSettings,
                 feature_map: dict[IntEnum, type[BaseScalarFeature]] = FEATURE_MAP) -> None:
        self._track_id: int = track_id
        self._board: HasFrames = board
        self.settings: DataLayerSettingsProtocol = cast(DataLayerSettingsProtocol, settings)
        self._plot: FeaturePlot[Frame] = FeaturePlot(LayerMode.FRAME, self.settings, color_settings, feature_map,
                                                     draw_labels=False, font_size=14)
        self._shader: Feature = Feature()

    @property
    def texture(self) -> Texture:
        return self._plot.fbo

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._plot.allocate(width, height, internal_format)
        self._shader.allocate()

    def deallocate(self) -> None:
        self._plot.deallocate()
        self._shader.deallocate()

    def clear(self) -> None:
        """Clear cached data."""
        self._plot.clear()

    def draw(self) -> None:
        self._plot.draw()

    def update(self) -> None:
        """Update visualization from the board's Frame."""
        if not self._plot.update_active():
            return
        pose: Frame | None = self._board.get_frame(self.settings.stage, self._track_id)
        cache = self._plot.data_cache
        cache.set(pose)

        fbo = self._plot.fbo
        if cache.lost:
            fbo.clear()

        if cache.idle or pose is None:
            return

        # Extract feature from frame
        feature_type = self._plot.feature_map[self.settings.feature_field]
        feature = pose[feature_type]
        offset = self.settings.angle_offset
        if offset != AngleOffset.NONE and has_pi_range(feature_type):
            feature = feature_type(offset_angles(feature.values, offset), feature.scores)

        # If showing angles, also fetch velocity for thickness modulation. Compare the
        # resolved feature type (not the enum member) so any feature_map/enum works.
        deltas = pose[AngleVelocity].values if feature_type is Angles else None

        line_width = 1.0 / fbo.height * self.settings.line_width
        line_smooth = 1.0 / fbo.height * self.settings.line_smooth
        display_range = feature.display_range()

        fbo.begin()
        clear_color()
        self._shader.use(feature, self._plot.colors(), line_width, line_smooth, self.settings.use_scores, display_range, deltas=deltas)
        fbo.end()

        if self.settings.render_labels:
            joint_enum_type = feature.__class__.enum()
            self._plot.update_labels([joint_enum_type(i).name for i in range(len(feature))], _label_place)
