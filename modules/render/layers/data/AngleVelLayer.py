# Standard library imports
from typing import Tuple

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub, DataHubType, PoseDataHubTypes
from modules.gl import Fbo, Texture, Blit, clear_color, draw_box_string, text_init
from modules.pose.Frame import Frame, FrameField
from modules.render.layers.LayerBase import LayerBase, DataCache, Rect
from modules.render.shaders import AngleVelShader
from .Colors import ANGLES_COLORS

from modules.utils.HotReloadMethods import HotReloadMethods


class AngleVelLayer(LayerBase):
    """Visualizes angles with line thickness modulated by angular velocity.

    Unique visualization: horizontal lines show angle positions, but line thickness
    dynamically changes based on angular velocity magnitude - high velocity = thick lines.
    """

    def __init__(self, track_id: int, data_hub: DataHub, data_type: PoseDataHubTypes,
                 line_thickness: float = 1.0, line_smooth: float = 1.0,
                 display_range: Tuple[float, float] | None = None) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._data_type: PoseDataHubTypes = data_type
        self._display_range: Tuple[float, float] | None = display_range

        self._fbo: Fbo = Fbo()
        self._label_fbo: Fbo = Fbo()
        self._data_cache: DataCache[Frame] = DataCache[Frame]()
        self._labels: list[str] = []

        self.line_thickness: float = line_thickness
        self.line_smooth: float = line_smooth
        self.draw_labels: bool = True

        self._shader: AngleVelShader = AngleVelShader()

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
            if self.draw_labels:
                Blit.use(self._label_fbo.texture)

    def update(self) -> None:
        """Update visualization from DataHub Frame."""
        pose: Frame | None = self._data_hub.get_item(DataHubType(self._data_type), self._track_id)
        self._data_cache.update(pose)

        if self._data_cache.lost:
            self._fbo.clear()

        if self._data_cache.idle or pose is None:
            return

        # Extract features from frame
        angles = pose.angles
        velocity = pose.angle_vel

        # Override display range if configured
        if self._display_range is not None:
            original_range = angles.range()
            angles._range = self._display_range  # type: ignore

        # Apply alpha to colors
        colors_with_alpha = [
            (r, g, b, a * 1.0)
            for r, g, b, a in ANGLES_COLORS
        ]

        line_thickness = 1.0 / self._fbo.height * self.line_thickness
        line_smooth = 1.0 / self._fbo.height * self.line_smooth

        self._fbo.begin()
        clear_color()
        self._shader.use(angles, velocity, line_thickness, line_smooth,
                        colors_with_alpha[0], colors_with_alpha[1])
        self._fbo.end()

        # Restore original range
        if self._display_range is not None:
            angles._range = original_range  # type: ignore

        # Render labels if changed
        joint_enum_type = angles.__class__.enum()
        num_joints: int = len(angles)
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
