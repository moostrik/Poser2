# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.DataHub import DataHub
from modules.gl import Fbo, Texture, Blit, clear_color, Text
from modules.pose.Frame import Frame, ScalarFrameField
from modules.render.layers.LayerBase import LayerBase, DataCache, Rect
from modules.render.shaders import AngleVelShader
from modules.render.layers.data.DataLayerConfig import FEATURE_COLORS, DEFAULT_COLORS, DataLayerConfig

from modules.utils.HotReloadMethods import HotReloadMethods


class AngleVelLayer(LayerBase):
    """Visualizes angles with line thickness modulated by angular velocity.

    Unique visualization: horizontal lines show angle positions, but line thickness
    dynamically changes based on angular velocity magnitude - high velocity = thick lines.
    """

    def __init__(self, track_id: int, data_hub: DataHub, config: DataLayerConfig) -> None:
        self._track_id: int = track_id
        self._data_hub: DataHub = data_hub
        self._config: DataLayerConfig = config
        self.active: bool = False  # Instance-level active state

        self._fbo: Fbo = Fbo()
        self._label_fbo: Fbo = Fbo()
        self._data_cache: DataCache[Frame] = DataCache[Frame]()
        self._labels: list[str] = []

        self.draw_labels: bool = False

        self._shader: AngleVelShader = AngleVelShader()
        self._text_renderer: Text = Text()

        self._hot_reloader = HotReloadMethods(self.__class__, True, True)

    def set_active(self, active: bool) -> None:
        """Set active state and trigger cleanup on deactivation."""
        if self.active != active:
            self.active = active
            if not active:
                self.clear()

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
        if not self.active:
            return
        if self._fbo.allocated:
            Blit.use(self._fbo.texture)
            if self.draw_labels and self._config.render_labels:
                Blit.use(self._label_fbo.texture)

    def update(self) -> None:
        """Update visualization from DataHub Frame."""
        if not self.active:
            return
        pose: Frame | None = self._data_hub.get_pose(self._config.stage, self._track_id)
        self._data_cache.update(pose)

        if self._data_cache.lost:
            self._fbo.clear()

        if self._data_cache.idle or pose is None:
            return

        # Extract features from frame
        angles = pose.angles
        velocity = pose.angle_vel

        # Use config colors or fallback to FEATURE_COLORS (angles)
        colors = self._config.colors or FEATURE_COLORS.get(ScalarFrameField.angles, DEFAULT_COLORS)

        line_width = 1.0 / self._fbo.height * self._config.line_width
        line_smooth = 1.0 / self._fbo.height * self._config.line_smooth
        display_range = angles.display_range()

        self._fbo.begin()
        clear_color()
        self._shader.use(angles, velocity, line_width, line_smooth, colors, display_range)
        self._fbo.end()

        # Render labels if changed
        if self._config.render_labels:
            joint_enum_type = angles.__class__.enum()
            num_joints: int = len(angles)
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

        colors = self._config.colors or FEATURE_COLORS.get(ScalarFrameField.angles, DEFAULT_COLORS)
        # Ensure we have at least one color
        if not colors:
            colors = DEFAULT_COLORS

        for i in range(num_labels):
            string: str = labels[i]
            x: int = int(rect.x + (i + 0.1) * step)
            y: int = int(rect.y + rect.height * 0.5 - 7)
            clr: int = i % len(colors)

            self._text_renderer.draw_box_text(
                x, y, string, colors[clr], (0.0, 0.0, 0.0, 0.3),
                screen_width=fbo.width, screen_height=fbo.height
            )

        fbo.end()
