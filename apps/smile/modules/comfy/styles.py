"""Style cycling, as aiia3's 'smile' keyword: one style for N visitors, then the next.

The style id is also the movement id for the music (files/music_map.json).
"""

from modules.comfy.settings import StyleSettings

# movement id -> the StyleSettings field holding its text
STYLE_FIELDS: dict[str, str] = {'0000': 'style_a', '0001': 'style_b'}


class Styles():
    def __init__(self, settings: StyleSettings) -> None:
        self.settings: StyleSettings = settings
        self._ids: list[str] = list(STYLE_FIELDS)
        self._index: int = 0
        self._count: int = 0

    def advance(self) -> str:
        """Call once per visitor; returns the style id for this visitor."""
        if self._count >= self.settings.repeat:
            self._count = 0
            self._index = (self._index + 1) % len(self._ids)
        self._count += 1
        self.settings.current = self.current_id()
        return self.current_id()

    def current_id(self) -> str:
        return self._ids[self._index]

    def current_text(self) -> str:
        return getattr(self.settings, STYLE_FIELDS[self.current_id()])
