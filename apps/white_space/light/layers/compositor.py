"""Compositor — the timeless look mixer.

Renders the look it was handed this tick: a weighted list of layers. Each entry's layer
draws itself into a private scratch frame (via its own blend mode) and is added
``weight ×`` into the output frame — the spun-content layers ring-rolled by ``light_phase``.

The Compositor holds no timing, easing, transition, or reset logic: weight curves live in
the show state classes, and layer resets are explicit (``reset_layers``). The one policy it
owns is the manual/debug override: while ``light.manual`` is on, the operator's
``manual_layers`` checklist (full weight each) replaces the state's entries.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ._base_layer import BaseLayer
from ..clock import Tick
from ..frame import Frame

if TYPE_CHECKING:
    from ..settings import LightSettings, LayerId

logger = logging.getLogger(__name__)

# The look: which layers to draw this tick, at which weight.
Look = list[tuple['LayerId', float]]


class Compositor:
    """Mixes the current look into the output frame; see the module docstring.

    Deliberately not a ``BaseLayer``: "layer" means exactly "a thing a state can put in
    its look" — the Compositor is the mixer those layers pass through.
    """

    def __init__(self, config: LightSettings, layers: dict[LayerId, BaseLayer],
                 shifted: set[LayerId]) -> None:
        self._config = config
        self._layers = layers
        self._shifted = shifted            # layers that get the light_phase ring shift
        self._entries: Look = []
        self._scratch = Frame(config.light_resolution, Tick(0.0, 0.0, 0.0, 0.0, 0))

    # -- Look input (light thread, same thread as render) ----------------------

    def set_look(self, entries: Look) -> None:
        """Set the look to draw this tick — called every tick by the state machine.
        A weight of 0.0 keeps the layer in the look but silent; layers are never
        reset implicitly."""
        self._entries = entries

    def reset_layers(self, ids: list[LayerId]) -> None:
        """Reset the named layers' internal state — called by a show state in ``enter()``
        when it wants a fresh start (resets are never automatic)."""
        for id in ids:
            layer = self._layers.get(id)
            if layer is not None:
                layer.reset()

    # -- Per-tick mix ----------------------------------------------------------

    def render(self, frame: Frame) -> None:
        cfg = self._config
        entries: Look = self._entries
        if cfg.manual:
            entries = [(id, 1.0) for id in cfg.manual_layers]

        shift = int(round(cfg.light_phase * frame.white.shape[0])) % frame.white.shape[0]

        s = self._scratch
        s.tick, s.motor, s.playhead = frame.tick, frame.motor, frame.playhead
        for id, weight in entries:
            if weight <= 0.0:
                continue
            layer = self._layers.get(id)
            if layer is None:
                continue
            s.light_img.fill(0.0)
            try:
                layer.render(s)   # blends into the private scratch via its own blend mode
            except Exception:
                logger.exception("Error in %s.render", layer.__class__.__name__)
                continue
            sw, sb = s.white, s.blue
            if shift and id in self._shifted:
                sw, sb = np.roll(sw, shift), np.roll(sb, shift)
            frame.white += weight * sw
            frame.blue  += weight * sb
