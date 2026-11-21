"""ScreenSize: the recording size presets, and the size rules: width and height are the truth; the preset and portrait
follow them."""

from enum import Enum


class ScreenSize(Enum):
    """A recording size preset, landscape (width, height); portrait swaps them."""
    HD =     (1920, 1080)
    QHD =    (2560, 1440)
    UHD =    (3840, 2160)
    CUSTOM = (0, 0)


def even(value: int) -> int:
    """`value` rounded down to an even number of at least 2, as NV12 needs."""
    return max(2, value - value % 2)


def preset_size(size: ScreenSize, portrait: bool) -> tuple[int, int]:
    """(width, height) of a preset, upright when `portrait`."""
    width, height = size.value
    return (height, width) if portrait else (width, height)


def matching_preset(width: int, height: int) -> ScreenSize:
    """The preset of this size in either orientation, else CUSTOM."""
    landscape = (max(width, height), min(width, height))
    for size in ScreenSize:
        if size is not ScreenSize.CUSTOM and size.value == landscape:
            return size
    return ScreenSize.CUSTOM
