"""What the clip player drives: one wall of the sphere, its cells by number (`render.layers.WallCompositor` meets it)."""

from typing import Protocol

import numpy as np


class Wall(Protocol):
    """One wall's cells, each by its number (0 to 23), from any thread."""

    def set_image(self, id: int, image: np.ndarray, initial: bool) -> None:
        """A cell's next clip frame; `initial` for the first frame of a play."""
        ...

    def set_caption(self, id: int, line: str) -> None:
        """A cell's caption line from now on."""
        ...

    def light(self, id: int, seconds: float, number: int = 0) -> None:
        """A cell lit white with its screen's `number` (0: none) for `seconds`."""
        ...

    def scroll_loop(self, name: str, speed: float, gap: float, size: int, ring: int, started: float,
                    cells: list[tuple[int, int]], last: int | None = None) -> bool:
        """A word going round a ring of cells; True for a loop that is new."""
        ...
