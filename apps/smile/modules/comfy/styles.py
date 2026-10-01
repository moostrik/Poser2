"""Style cycling, as aiia3's 'smile' keyword: one style for N visitors, then the next.

The style id is also the movement id for the music (data/audio/music_map.json).
"""

from modules.comfy.settings import StyleSettings

# movement id -> the letter of its StyleSettings fields (style_a, negative_a, ...)
STYLE_LETTERS: dict[str, str] = {'0000': 'a', '0001': 'b'}


class Styles():
    def __init__(self, settings: StyleSettings) -> None:
        self.settings: StyleSettings = settings
        self._ids: list[str] = list(STYLE_LETTERS)
        self._index: int = 0
        self._count: int = 0

    def advance(self) -> str:
        """Call once per visitor; returns the style id for this visitor."""
        if self._count >= self.settings.repeat:
            self._count = 0
            self._index = (self._index + 1) % len(self._ids)
        self._count += 1
        self.settings.current = STYLE_LETTERS[self.current_id()].upper()
        return self.current_id()

    def current_id(self) -> str:
        return self._ids[self._index]

    def current_text(self) -> str:
        return getattr(self.settings, f'style_{STYLE_LETTERS[self.current_id()]}')

    def current_negative(self) -> str:
        return getattr(self.settings, f'negative_{STYLE_LETTERS[self.current_id()]}')
