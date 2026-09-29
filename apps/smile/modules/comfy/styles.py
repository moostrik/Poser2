"""Style cycling, as aiia3's 'smile' keyword: one style for N visitors, then the next.

The style id is also the movement id for the music (files/music_map.json).
"""

import json


class Styles():
    def __init__(self, path: str) -> None:
        with open(path, encoding='utf-8') as f:
            data: dict = json.load(f)
        self._ids: list[str] = list(data.keys())
        self._texts: dict[str, str] = {key: entry['name'] for key, entry in data.items()}
        self._index: int = 0
        self._count: int = 0

    def advance(self, repeat: int) -> str:
        """Call once per visitor; returns the style id for this visitor."""
        if self._count >= repeat:
            self._count = 0
            self._index = (self._index + 1) % len(self._ids)
        self._count += 1
        return self.current_id()

    def current_id(self) -> str:
        return self._ids[self._index]

    def current_text(self) -> str:
        return self._texts[self.current_id()]
