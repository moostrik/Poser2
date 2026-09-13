"""The beam lights on the walls — the pure geometry behind the render's beam-mode simulation.

No GL here, so it is unit-testable on its own; ``BeamLightSimulationLayer`` wraps it for the
screen.

A pixel index is an azimuth: the front white points along the playhead heading and the other
three lights sit at ``BEAM_LIGHT_HEADINGS`` from it. Each lit light is a line of ``width``
degrees at its level, fading to nothing over ``blur`` degrees on each side — a raised cosine,
so the edge meets both the solid core and the darkness around it without a kink.

Recent flashes (``Flash`` on the board) are drawn the same way at the heading they lit, fading
out over ``flash_seconds`` — a display aid, so a flash the fixture shows for a tick or two stays
readable on screen.
"""

import math

import numpy as np

from modules.board import Flash

from apps.white_space.light import BeamLightId, BEAM_LIGHT_CHANNEL, BEAM_LIGHT_HEADINGS
from apps.white_space.light.layers import BlendType, normalize_azimuth, apply_circular


def beam_profile(width: float, blur: float, resolution: int) -> np.ndarray:
    """The symmetric line profile in pixels: 1.0 across a ``width``-wide core, fading to 0.0
    over ``blur`` on each side (both angles in radians; ``blur`` 0 = a hard edge).

    Always odd-length with a single centre pixel — integer pixel geometry throughout, so a
    line's centre never lands a pixel off — and never longer than the full turn.
    """
    span_max: int = (resolution - 1) // 2
    core: int = min(max(0, round(width / 2.0 / math.tau * resolution)), span_max)
    fade: int = min(max(0, round(blur / math.tau * resolution)), span_max - core)
    distance = np.abs(np.arange(-(core + fade), core + fade + 1))
    if fade == 0:
        return (distance <= core).astype(np.float64)
    return 0.5 + 0.5 * np.cos(math.pi * np.clip((distance - core) / fade, 0.0, 1.0))


def _paint_beam_lights(beam_lights: np.ndarray, heading: float, profile: np.ndarray,
                       out: np.ndarray) -> None:
    """MAX-blend the four beam lights at ``heading`` into ``out``, without clearing it."""
    resolution: int = out.shape[1]
    half: int = len(profile) // 2
    for light in BeamLightId:
        level = float(beam_lights[light])
        if not level > 0.0:                                            # NaN and ≤ 0 paint nothing
            continue
        centre = round(normalize_azimuth(heading + BEAM_LIGHT_HEADINGS[light]) * resolution)
        channel = out[0, :, BEAM_LIGHT_CHANNEL[light]]
        apply_circular(channel, (level * profile).astype(out.dtype), centre - half, BlendType.MAX)


def project_beam_lights(beam_lights: np.ndarray, heading: float, width: float, blur: float,
                        out: np.ndarray) -> None:
    """Paint the beam lights into ``out`` (shape ``(1, R, 3)``, the Frame layout; zeroed first).

    ``beam_lights`` are the four levels (index = ``BeamLightId``), ``heading`` the front white's
    direction, ``width`` and ``blur`` the line's core width and edge falloff — all in radians.
    Channel 0 = white, 1 = blue.
    """
    out.fill(0.0)
    _paint_beam_lights(beam_lights, heading, beam_profile(width, blur, out.shape[1]), out)


def paint_flashes(flashes: list[Flash], now: float, seconds: float, width: float, blur: float,
                  out: np.ndarray) -> None:
    """MAX-blend the flashes younger than ``seconds`` into ``out``, each at its own heading and
    faded linearly by its age (full when new, gone at ``seconds``; ``seconds`` 0 draws none).
    A flash's white is the front white lamp, its blue both blue lamps — as ``beam_flash`` lights
    them. ``now`` is on the flashes' monotonic clock; angles in radians.
    """
    if seconds <= 0.0:
        return
    profile = beam_profile(width, blur, out.shape[1])
    levels = np.zeros(len(BeamLightId), dtype=out.dtype)
    for flash in flashes:
        fade = min(1.0, 1.0 - (now - flash.timestamp) / seconds)   # posted after `now` reads as new
        if fade <= 0.0:
            continue
        levels[BeamLightId.FRONT_WHITE] = flash.white * fade
        levels[BeamLightId.LEFT_BLUE] = levels[BeamLightId.RIGHT_BLUE] = flash.blue * fade
        _paint_beam_lights(levels, flash.azimuth, profile, out)
