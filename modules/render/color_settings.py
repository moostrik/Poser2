"""ColorSettings — reactive per-player track colors.

Extracted into its own module to avoid circular imports between
render_settings.py (which imports layer Settings classes) and the
layer modules (which need ColorSettings at runtime).
"""

from modules.settings import Field, BaseSettings
from modules.utils import Color


class ColorSettings(BaseSettings):
    """Per-player track colors (one color picker per player in the panel)."""
    player_0:    Field[Color] = Field(Color(1.0, 0.0, 0.0))  # Red
    player_1:    Field[Color] = Field(Color(0.0, 0.0, 1.0))  # Blue
    player_2:    Field[Color] = Field(Color(0.0, 1.0, 0.0))  # Green
    pose_left:   Field[Color] = Field(Color(1.0, 0.5, 0.0))  # Orange
    pose_right:  Field[Color] = Field(Color(0.0, 1.0, 1.0))  # Cyan
    pose_center: Field[Color] = Field(Color(1.0, 1.0, 1.0))  # White
    history:     Field[Color] = Field(Color(0.5, 0.5, 0.5))  # Grey

    @property
    def track_colors(self) -> list[Color]:
        """Return all player colors as an ordered list."""
        return [v for k, v in sorted(
            ((n, getattr(self, n)) for n in self._fields if n.startswith('player_')),
            key=lambda pair: pair[0],
        )]

    @property
    def track_color_tuples(self) -> list[tuple[float, float, float, float]]:
        """Return all player colors as RGBA tuples (for GL layers)."""
        return [c.to_tuple() for c in self.track_colors]

    @property
    def default_color_tuples(self) -> list[tuple[float, float, float, float]]:
        """Default data-layer palette (pose_left + pose_right) as RGBA tuples."""
        return [self.pose_left.to_tuple(), self.pose_right.to_tuple()]
