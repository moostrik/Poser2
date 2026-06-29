"""Crossfade — the speed compositor, itself a layer.

The CPU mirror of ``modules.render`` ``CompositeLayer``: a layer that is handed the slot layer pools
and owns the combining. Each tick it reads the motor rpm off the frame, computes the idle/low/high
crossfade weights, resolves each slot's selected layers from its pool, blends them (via their own
blend modes) into a private scratch, and adds ``weight ×`` that composite into the output — the high
slot ring-rolled by ``light_phase``. The renderer just builds the pools and calls ``render(frame)``.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from ._base_layer import BaseLayer, LayerSettings
from ..clock import Tick
from ..frame import Frame

if TYPE_CHECKING:
    from ...board import Board
    from ..settings import LightSettings, LowLayerId, HighLayerId

logger = logging.getLogger(__name__)

# Crossfade dead-zone: each fade starts `deadzone` ABOVE the mode below it and ends `deadzone` BELOW
# its own mode rpm, so a band around every mode rpm shows only that mode (no neighbour bleed on jitter).
CROSSFADE_DEADZONE: float = 0.1


def _ease(t: float) -> float:
    """Sine ease-in-out on [0,1] — the crossfade accelerates out of one comp and settles into the next."""
    return 0.5 - 0.5 * math.cos(math.pi * t)


def crossfade_weights(rpm: float, idle_rpm: float, low_rpm: float,
                      high_cross_rpm: float, deadzone: float = CROSSFADE_DEADZONE) -> tuple[float, float, float]:
    """``(w_idle, w_low, w_high)`` for the motor rpm. Each comp fades in over a band *inside* the gap
    between mode rpms — from a deadzone above the mode below it to a deadzone below its own mode rpm —
    so the dead zone around each mode rpm keeps the neighbouring comp off under rpm jitter (e.g. in low
    mode the high comp stays fully off). The fade is sine-eased in/out. High ends at ``high_cross_rpm``
    (nothing higher to bleed). Weights sum to 1."""
    eps = 1e-6
    s = min(1.0, max(0.0, (rpm - idle_rpm * (1.0 + deadzone)) / max(low_rpm * (1.0 - deadzone) - idle_rpm * (1.0 + deadzone), eps)))
    h = min(1.0, max(0.0, (rpm - low_rpm * (1.0 + deadzone)) / max(high_cross_rpm - low_rpm * (1.0 + deadzone), eps)))
    s_in, h_in = _ease(s), _ease(h)
    return (1.0 - s_in) * (1.0 - h_in), s_in * (1.0 - h_in), h_in


class Crossfade(BaseLayer):
    """Speed compositor: blends each slot's selected layers and crossfades the slots by motor rpm.

    Handed the low/high layer pools (built by the renderer, like the GPU ``CompositeLayer(layers, …)``).
    Uses the ``BaseLayer`` template: ``_draw`` accumulates the weighted composite into this layer's own
    scratch, then ``render`` ADD-blends it into the frame. Owns reset-on-deselect: a layer that drops
    out of every slot gets ``reset()`` so it restarts fresh on reselection.
    """

    def __init__(self, config: LightSettings,
                 low_layers: dict[LowLayerId, BaseLayer],
                 high_layers: dict[HighLayerId, BaseLayer], board: Board) -> None:
        super().__init__(config.light_resolution, LayerSettings(), board)   # blend=ADD: own scratch → frame
        self._config = config
        self._low_layers = low_layers
        self._high_layers = high_layers
        self._scratch = Frame(config.light_resolution, Tick(0.0, 0.0, 0.0, 0.0, 0))
        self._prev: set[BaseLayer] = set()

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        cfg = self._config
        # Effective speed: the measured rpm below the sensor ceiling, the commanded speed above it
        # (the sensor can't see past ~200 rpm), 0 when stopped → resolves to the idle slot.
        rpm = frame.motor.effective_rpm
        w_idle, w_low, w_high = crossfade_weights(
            rpm, cfg.motor.idle_rpm, cfg.motor.low_rpm, cfg.high_cross_rpm)
        shift = int(round(cfg.light_phase * self.resolution)) % self.resolution

        pick = lambda d, ids: [d[i] for i in ids if i in d]   # selected instances, in blend order
        entries = (
            (pick(self._low_layers,  cfg.idle_layers), w_idle, 0),
            (pick(self._low_layers,  cfg.low_layers),  w_low,  0),
            (pick(self._high_layers, cfg.high_layers), w_high, shift),   # high slot gets light_phase
        )

        # Reset any layer no longer selected in any slot, so it restarts fresh on reselection.
        current = {layer for layers, _, _ in entries for layer in layers}
        for layer in self._prev - current:
            layer.reset()
        self._prev = current

        s = self._scratch
        s.tick, s.motor, s.playhead = frame.tick, frame.motor, frame.playhead
        for layers, weight, sh in entries:
            if weight <= 0.0 or not layers:
                continue
            s.light_img.fill(0.0)
            drew = False
            for layer in layers:
                try:
                    layer.render(s)   # blends into the private scratch via its own blend mode
                    drew = True
                except Exception:
                    logger.exception("Error in %s.render", layer.__class__.__name__)
            if not drew:
                continue
            sw, sb = s.white, s.blue
            if sh:
                sw, sb = np.roll(sw, sh), np.roll(sb, sh)
            white += weight * sw
            blue  += weight * sb

    def reset(self) -> None:
        for layer in set(self._low_layers.values()) | set(self._high_layers.values()):
            layer.reset()
        self._prev = set()
