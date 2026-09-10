"""The beam lights on the walls — the pure geometry behind the render's beam-mode simulation.

No GL here, so it is unit-testable on its own; ``BeamLightSimulationLayer`` wraps it for the
screen.

A strip index is an angle: the front white points along the playhead heading and the other
three lights sit at ``BEAM_LIGHT_HEADINGS`` from it. Each lit light is a line of ``width``
degrees at its level, fading to nothing over ``blur`` degrees on each side — a raised cosine,
so the edge meets both the solid core and the darkness around it without a kink.
"""

import math

import numpy as np

from apps.white_space.light import BeamLightId, BEAM_LIGHT_CHANNEL, BEAM_LIGHT_HEADINGS
from apps.white_space.light.layers import BlendType, angle_to_strip_position, apply_circular


def beam_profile(width: float, blur: float, resolution: int) -> np.ndarray:
    """The symmetric line profile in pixels: 1.0 across a ``width``-wide core, fading to 0.0
    over ``blur`` on each side (both angles in radians; ``blur`` 0 = a hard edge).

    Always odd-length with a single centre pixel — integer pixel geometry throughout, so a
    line's centre never lands a pixel off — and never longer than the strip.
    """
    span_max: int = (resolution - 1) // 2
    core: int = min(max(0, round(width / 2.0 / math.tau * resolution)), span_max)
    fade: int = min(max(0, round(blur / math.tau * resolution)), span_max - core)
    distance = np.abs(np.arange(-(core + fade), core + fade + 1))
    if fade == 0:
        return (distance <= core).astype(np.float64)
    return 0.5 + 0.5 * np.cos(math.pi * np.clip((distance - core) / fade, 0.0, 1.0))


def project_beam_lights(beam_lights: np.ndarray, heading: float, width: float, blur: float,
                        out: np.ndarray) -> None:
    """Paint the beam lights into ``out`` (shape ``(1, R, 3)``, the Frame layout; zeroed first).

    ``beam_lights`` are the four levels (index = ``BeamLightId``), ``heading`` the front white's
    direction, ``width`` and ``blur`` the line's core width and edge falloff — all in radians.
    Channel 0 = white, 1 = blue.
    """
    out.fill(0.0)
    resolution: int = out.shape[1]
    profile = beam_profile(width, blur, resolution)
    half: int = len(profile) // 2
    for light in BeamLightId:
        level = float(beam_lights[light])
        if not level > 0.0:                                            # NaN and ≤ 0 paint nothing
            continue
        centre = round(angle_to_strip_position(heading + BEAM_LIGHT_HEADINGS[light]) * resolution)
        channel = out[0, :, BEAM_LIGHT_CHANNEL[light]]
        apply_circular(channel, (level * profile).astype(out.dtype), centre - half, BlendType.MAX)
