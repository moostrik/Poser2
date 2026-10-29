"""Style cycling, as aiia3's 'smile' keyword: one style for N visitors, then the next; or one style
for everyone (style_mode).

The style id also picks the music (data/audio/music_map.json).
"""

from .settings import GeneratorSettings, StyleMode, StyleSettings

# style id -> the letter of its StyleSettings group (style_a, style_b)
STYLE_LETTERS: dict[str, str] = {'0000': 'a', '0001': 'b'}
# style_mode A or B: the index of the one style every visitor gets
FORCED: dict[StyleMode, int] = {StyleMode.A: 0, StyleMode.B: 1}


class Styles():
    def __init__(self, settings: GeneratorSettings) -> None:
        self.settings: GeneratorSettings = settings
        self._ids: list[str] = list(STYLE_LETTERS)
        self._index: int = 0
        self._count: int = 0

    def advance(self) -> str:
        """Call once per visitor; returns the style id for this visitor."""
        mode: StyleMode = self.settings.style_mode
        if mode is StyleMode.BOTH:
            if self._count >= self._visitors():
                self._count = 0
                self._index = (self._index + 1) % len(self._ids)
            self._count += 1
        else:
            # forced; back on Both this style starts a full run
            self._index = FORCED[mode]
            self._count = 0
        self.settings.style_current = STYLE_LETTERS[self.current_id()].upper()
        return self.current_id()

    def _visitors(self) -> int:
        """How many visitors the current style gets before the other one (each style has its own number)."""
        return getattr(self.settings, f'style_visitors_{STYLE_LETTERS[self.current_id()]}')

    def current_id(self) -> str:
        return self._ids[self._index]

    def current(self) -> StyleSettings:
        return getattr(self.settings, f'style_{STYLE_LETTERS[self.current_id()]}')
