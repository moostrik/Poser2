"""ColorSettings: the players' track colors, the pose's side colors and the history color."""

from modules.settings import Field, BaseSettings
from modules.utils import Color


class ColorSettings(BaseSettings):
    """Per-player track colors (one color picker per player in the panel)."""
    player_0:    Field[Color] = Field(Color(1.0, 0.0, 0.0), description="Player 0's track color")
    player_1:    Field[Color] = Field(Color(0.0, 0.0, 1.0), description="Player 1's track color")
    player_2:    Field[Color] = Field(Color(0.0, 1.0, 0.0), description="Player 2's track color")
    player_3:    Field[Color] = Field(Color(1.0, 1.0, 0.0), description="Player 3's track color")
    player_4:    Field[Color] = Field(Color(1.0, 0.0, 1.0), description="Player 4's track color")
    player_5:    Field[Color] = Field(Color(0.0, 1.0, 1.0), description="Player 5's track color")
    player_6:    Field[Color] = Field(Color(1.0, 0.5, 0.0), description="Player 6's track color")
    player_7:    Field[Color] = Field(Color(0.5, 0.0, 1.0), description="Player 7's track color")
    pose_left:   Field[Color] = Field(Color(1.0, 0.5, 0.0), description="The pose's left side")
    pose_right:  Field[Color] = Field(Color(0.0, 1.0, 1.0), description="The pose's right side")
    pose_center: Field[Color] = Field(Color(1.0, 1.0, 1.0), description="The pose's center line")
    history:     Field[Color] = Field(Color(0.5, 0.5, 0.5), description="A feature drawn in one color, as its history")

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
