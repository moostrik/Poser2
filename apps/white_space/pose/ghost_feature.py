"""GhostFeature — the White Space per-pose ghost metrics: Dwell, Motion, Stability, Fade (each in [0, 1]).

The Ghoster stamps this on every pose it emits — live poses and ghosts alike:

- ``Dwell``     — how long the pose has held its spot (sweeps on the spot / ``dwell_sweeps``).
- ``Motion``    — on-spot performance (accumulated ``MotionTime`` / ``motion_scale``).
- ``Stability`` — how held the pose is (consecutive similar sweeps / ``stability_sweeps``); 1.0 = valid.
- ``Fade``      — how present the pose is: ``1.0`` live / fresh; a released ghost decays it 1→0.

Kinds are told apart by the separate ``GhostState`` feature (present only on ghosts); this feature is
pure metrics. It replaces the Ghoster's old use of ``PlayheadStability`` (dwell/motion) and the
standalone ``Fade`` feature. Playhead is a White Space concept, so it lives with the app.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from modules.pose.features import NormalizedScalarFeature


class GhostElement(IntEnum):
    """The per-pose ghost metrics carried by ``GhostFeature``."""
    Dwell     = 0
    Motion    = 1
    Stability = 2
    Fade      = 3


class GhostFeature(NormalizedScalarFeature[GhostElement]):
    """Per-pose ``Dwell`` / ``Motion`` / ``Stability`` / ``Fade``, each in ``[0, 1]``. Absent (NaN, score
    0) before the Ghoster has stamped it."""

    @classmethod
    def enum(cls) -> type[GhostElement]:
        return GhostElement

    @classmethod
    def make(cls, dwell: float, motion: float, stability: float, fade: float) -> "GhostFeature":
        """Build from the four metrics (all valid)."""
        return cls(values=np.array([dwell, motion, stability, fade], dtype=np.float32),
                   scores=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
