"""Patch: the square of pixels or samples a window or a median spans, by its radius."""

from enum import IntEnum


class Patch(IntEnum):
    """A square patch; its value is the radius, so a shader takes it as is. ONE is the pixel alone."""
    ONE = 0
    THREE = 1
    FIVE = 2
    SEVEN = 3
    NINE = 4

    @property
    def label(self) -> str:
        """The patch's size as the panel shows it, 1×1 to 9×9."""
        side = 2 * int(self) + 1
        return f'{side}×{side}'
