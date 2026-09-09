"""The bar's lights on the walls — the pure geometry behind the render's low-speed simulation.

No GL here, so it is unit-testable on its own; ``BarLightSimulationLayer`` wraps it for the
screen.

A strip index is an angle: the front white points along the playhead heading and the other
three lights sit at ``BAR_LIGHT_HEADINGS`` from it. Each lit light is a line of ``beam``
degrees at its level, fading to nothing over ``blur`` degrees on each side — a raised cosine,
so the edge meets both the solid core and the darkness around it without a kink.
"""

import math

import numpy as np

from apps.white_space.light import BarLightId, BAR_LIGHT_CHANNEL, BAR_LIGHT_HEADINGS
from apps.white_space.light.layers import BlendType, angle_to_strip_position, apply_circular


def beam_profile(beam: float, blur: float, resolution: int) -> np.ndarray:
    """The symmetric line profile in pixels: 1.0 across a ``beam``-wide core, fading to 0.0
    over ``blur`` on each side (both angles in radians; ``blur`` 0 = a hard edge).

    Always odd-length with a single centre pixel — integer pixel geometry throughout, so a
    line's centre never lands a pixel off — and never longer than the strip.
    """
    span_max: int = (resolution - 1) // 2
    core: int = min(max(0, round(beam / 2.0 / math.tau * resolution)), span_max)
    fade: int = min(max(0, round(blur / math.tau * resolution)), span_max - core)
    distance = np.abs(np.arange(-(core + fade), core + fade + 1))
    if fade == 0:
        return (distance <= core).astype(np.float64)
    return 0.5 + 0.5 * np.cos(math.pi * np.clip((distance - core) / fade, 0.0, 1.0))


def project_bar_lights(bar_lights: np.ndarray, heading: float, beam: float, blur: float,
                       out: np.ndarray) -> None:
    """Paint the bar lights into ``out`` (shape ``(1, R, 3)``, the Frame layout; zeroed first).

    ``bar_lights`` are the four levels (index = ``BarLightId``), ``heading`` the front white's
    direction, ``beam`` and ``blur`` the line's core width and edge falloff — all in radians.
    Channel 0 = white, 1 = blue.
    """
    out.fill(0.0)
    resolution: int = out.shape[1]
    profile = beam_profile(beam, blur, resolution)
    half: int = len(profile) // 2
    for light in BarLightId:
        level = float(bar_lights[light])
        if not level > 0.0:                                            # NaN and ≤ 0 paint nothing
            continue
        centre = round(angle_to_strip_position(heading + BAR_LIGHT_HEADINGS[light]) * resolution)
        channel = out[0, :, BAR_LIGHT_CHANNEL[light]]
        apply_circular(channel, (level * profile).astype(out.dtype), centre - half, BlendType.MAX)
