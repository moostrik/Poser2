"""Crossfade — the mode compositor, itself a layer.

The CPU mirror of ``modules.render`` ``CompositeLayer``: a layer that is handed the slot layer pools
and owns the combining. Each tick it resolves the motor *mode* to a slot (idle/low/high), time-fades
the idle/low/high crossfade weights toward that slot — with separate up/down durations — resolves each
slot's selected layers from its pool, blends them (via their own blend modes) into a private scratch,
and adds ``weight ×`` that composite into the output — the high slot ring-rolled by ``light_phase``.
The renderer just builds the pools and calls ``render(frame)``.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from ._base_layer import BaseLayer, LayerSettings
from ..clock import Tick
from ..frame import Frame
from ..motor import MotorMode

if TYPE_CHECKING:
    from ...board import Board
    from ..settings import LightSettings, LowLayerId, HighLayerId

logger = logging.getLogger(__name__)


def _ease(t: float) -> float:
    """Sine ease-in-out on [0,1] — the crossfade accelerates out of one slot and settles into the next."""
    return 0.5 - 0.5 * math.cos(math.pi * t)


def _slot_for_mode(mode: MotorMode) -> int:
    """Motor mode → crossfade slot index: idle(0) ← STOPPED/IDLE, low(1) ← LOW, high(2) ← HIGH."""
    if mode == MotorMode.LOW:  return 1
    if mode == MotorMode.HIGH: return 2
    return 0


def transition_weights(from_w: tuple[float, float, float], target_slot: int,
                       t: float) -> tuple[float, float, float]:
    """Sine-eased lerp from ``from_w`` toward one-hot(``target_slot``); ``t`` clamped to [0,1].
    ``from_w`` sums to 1 and one-hot sums to 1, so the result sums to 1."""
    e = _ease(max(0.0, min(1.0, t)))
    return (
        (1.0 - e) * from_w[0] + (e if target_slot == 0 else 0.0),
        (1.0 - e) * from_w[1] + (e if target_slot == 1 else 0.0),
        (1.0 - e) * from_w[2] + (e if target_slot == 2 else 0.0),
    )


class Crossfade(BaseLayer):
    """Mode compositor: blends each slot's selected layers and time-crossfades the slots by motor mode.

    Handed the low/high layer pools (built by the renderer, like the GPU ``CompositeLayer(layers, …)``).
    Uses the ``BaseLayer`` template: ``_draw`` accumulates the weighted composite into this layer's own
    scratch, then ``render`` ADD-blends it into the frame. Owns reset-on-deselect: a layer that drops
    out of every slot gets ``reset()`` so it restarts fresh on reselection.

    The slot weights are decoupled from rpm: the motor *mode* selects a slot, and the weights ease to
    that slot's one-hot target over ``fade_times`` seconds — the up time when climbing the slot order
    (idle→low→high), the down time when dropping back. At rest the weights sit one-hot on the slot.
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
        # Time-driven slot transition state (sentinel -1 → first frame snaps, no startup fade).
        self._target_slot: int = -1
        self._cur_w: tuple[float, float, float] = (1.0, 0.0, 0.0)
        self._from_w: tuple[float, float, float] = (1.0, 0.0, 0.0)
        self._t_start: float = 0.0
        self._duration: float = 0.0

    def _draw(self, frame: Frame, white: np.ndarray, blue: np.ndarray) -> None:
        cfg = self._config
        # The motor mode selects a slot; the weights ease toward it over the up/down fade time.
        now = frame.tick.time
        slot = _slot_for_mode(frame.motor.mode)
        if slot != self._target_slot:
            if self._target_slot < 0:          # first observation: snap, don't animate
                self._from_w = (1.0 if slot == 0 else 0.0, 1.0 if slot == 1 else 0.0, 1.0 if slot == 2 else 0.0)
                self._duration = 0.0
            else:
                self._from_w = self._cur_w     # start from the current on-screen blend
                up, down = (list(cfg.fade_times) + [0.0, 0.0])[:2]
                self._duration = up if slot > self._target_slot else down
            self._t_start = now
            self._target_slot = slot

        t = 1.0 if self._duration <= 0.0 else (now - self._t_start) / self._duration
        w_idle, w_low, w_high = self._cur_w = transition_weights(self._from_w, slot, t)
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
        self._target_slot = -1   # re-snap to the current mode's slot on the next draw
