"""The log axes of the strip and the histogram: where each decade's line sits in a lane, as a share
of the lane's height from its bottom, and what it reads. Plain numbers, so they test without GL; the
shaders `strip_roll` and `histogram_draw` draw the bars on the same axes."""

import math
from dataclasses import dataclass

STRIP_DECADES: int = 3          # a strip lane runs from 10^-3 to 1 of a mean times scale
LANE_FILL: float = 0.94         # the share of a lane a bar may fill; the rest is the gap between lanes
HISTOGRAM_SHARES: tuple[float, ...] = (0.0001, 0.001, 0.01, 0.1)    # the histogram's lines, shares of the samples


@dataclass(frozen=True)
class AxisLine:
    """One line of a log axis: its height in the lane (0 bottom, 1 top) and its label."""
    fraction: float
    label: str


def strip_height(mean: float, scale: float) -> float:
    """Where the strip draws a mean amount in its lane at `scale`, as `strip_roll.frag` does."""
    level: float = (math.log10(max(mean * scale, 1e-6)) + STRIP_DECADES) / STRIP_DECADES
    return min(max(level, 0.0), 1.0) * LANE_FILL


def strip_lines(scale: float) -> list[AxisLine]:
    """The round decades of the mean amount (0.001, 0.01, ...) that fall inside the strip's lane at
    `scale`, each where the strip draws it."""
    if scale <= 0.0:
        return []
    lowest: int = math.ceil(-STRIP_DECADES - math.log10(scale))
    lines: list[AxisLine] = []
    for exponent in range(lowest, lowest + STRIP_DECADES + 1):
        amount: float = 10.0 ** exponent
        level: float = (math.log10(amount * scale) + STRIP_DECADES) / STRIP_DECADES
        if 0.05 < level <= 1.0 + 1e-9:          # not at the very bottom, where its label would sit on the next lane
            lines.append(AxisLine(level * LANE_FILL, f'{amount:g}'))
    return lines


def histogram_lines(samples: int) -> list[AxisLine]:
    """The histogram's round shares of the samples, each where `histogram_draw.frag` draws a bin
    holding that share; a share under one sample has no line."""
    lines: list[AxisLine] = []
    for share in HISTOGRAM_SHARES:
        count: float = share * samples
        if count < 1.0:
            continue
        lines.append(AxisLine(LANE_FILL * math.log1p(count) / math.log1p(samples), f'{100.0 * share:g}%'))
    return lines


def value_limit(scale: float) -> str:
    """The value at the right edge of the histogram and of the colormap key, 1 after `scale`; the left edge is the same below 0."""
    return f'{1.0 / scale:.3g}' if scale > 0.0 else '-'
