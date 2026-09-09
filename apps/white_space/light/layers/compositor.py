"""Compositor — the timeless look mixer.

Renders the look it was handed this tick: a weighted list of layers. Each entry's layer
draws itself into a private scratch frame (via its own blend mode) and is added
``weight ×`` into the output frame — the spun-content layers ring-rolled by ``light_phase``.

The Compositor holds no timing, easing, transition, or reset logic: weight curves live in
the show state classes, and layer resets are explicit (``reset_layers``). The one policy it
owns is its half of the debug override: while ``light.debug`` selects a layer, that one
layer (solo, full weight) replaces the state's entries (the Conductor owns the other half —
the motor auto-following the selected layer's regime).
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

# The mix: which layers to draw this tick, at which weight. A plain float weighs both
# channels; a (white, blue) tuple weighs them independently (the show specifies white
# and blue light separately per state).
MixWeight = 'float | tuple[float, float]'
Mix = list[tuple['LayerId', 'float | tuple[float, float]']]


def _channel_weights(weight: 'float | tuple[float, float]') -> tuple[float, float]:
    """Normalize a mix weight to per-channel (white, blue)."""
    if isinstance(weight, tuple):
        return weight
    return (weight, weight)


class Compositor:
    """Mixes the current mix into the output frame; see the module docstring.

    Deliberately not a ``BaseLayer``: "layer" means exactly "a thing a state can put in
    its mix" — the Compositor is the mixer those layers pass through.
    """

    def __init__(self, config: LightSettings, layers: dict[LayerId, BaseLayer]) -> None:
        self._config = config
        self._layers = layers              # each layer's SHIFTED flag (HighLayer) grants the ring shift
        self._entries: Mix = []
        self._scratch = Frame(config.light_resolution, Tick(0.0, 0.0))

    # -- Mix input (light thread, same thread as render) ----------------------

    def set_mix(self, entries: Mix) -> None:
        """Set the mix to draw this tick — called every tick by the state machine.
        A weight of 0.0 keeps the layer in the mix but silent; layers are never
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
        entries: Mix = self._entries
        if int(cfg.debug) != 0:                       # a debug layer is selected → solo it
            from ..settings import LayerId            # local: settings imports this package
            entries = [(LayerId(int(cfg.debug)), 1.0)]

        shift = int(round(cfg.light_phase * frame.white.shape[0])) % frame.white.shape[0]

        s = self._scratch
        s.tick, s.motor, s.playhead = frame.tick, frame.motor, frame.playhead
        for id, weight in entries:
            w_white, w_blue = _channel_weights(weight)
            if w_white <= 0.0 and w_blue <= 0.0:
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
            if shift and layer.SHIFTED:
                sw, sb = np.roll(sw, shift), np.roll(sb, shift)
            if w_white > 0.0:
                frame.white += w_white * sw
            if w_blue > 0.0:
                frame.blue  += w_blue * sb
