"""ColorSettings — reactive per-player track colors.

Extracted into its own module to avoid circular imports between
render_settings.py (which imports layer Settings classes) and the
layer modules (which need ColorSettings at runtime).
"""

from modules.settings import Setting, BaseSettings
from modules.utils import Color


class ColorSettings(BaseSettings):
    """Per-player track colors (one color picker per player in the panel)."""
    player_0: Setting[Color] = Setting(Color(1.0, 0.0, 0.0))  # Red
    player_1: Setting[Color] = Setting(Color(0.0, 0.0, 1.0))  # Blue
    player_2: Setting[Color] = Setting(Color(0.0, 1.0, 0.0))  # Green
    history:  Setting[Color] = Setting(Color(0.5, 0.5, 0.5))   # Grey

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
