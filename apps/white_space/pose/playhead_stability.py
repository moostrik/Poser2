"""PlayheadStability — a White Space-native pose feature: how consistently a person holds
their pose across successive playhead sweeps. It measures the *pose's* consistency; the
playhead is only the sampling trigger.

Each time the rotating playhead crosses a person (a ``PlayheadOffset`` zero-crossing) the
current pose (``Angles``) is banked into a short per-person ring of past sweeps. Stability
is computed *every frame* as the summed similarity of the **live** pose to those banked
sweeps, normalised by ring capacity: ``0.0`` with no banked sweeps and ``1.0`` only when
the live pose matches a full ring. Because it scores the live pose — not the last sampled
one — the value tracks the person in real time as the playhead approaches, rather than
lagging a full rotation behind. The history (and the value) reset when the person's azimuth
drifts too far from where the history began.

Playhead is a White Space concept, so the feature, its extractor and its settings live with
the app rather than in ``modules/pose`` — like ``PlayheadOffset`` (see ``playhead_offset.py``).
The open Frame ECS still lets the feature ride on ``Frame`` (via ``replace``) without
modules depending on app code.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from modules.pose.features import Angles, Azimuth, NormalizedSingleValue
from modules.pose.frame import Frame, replace
from modules.pose.nodes import FilterNode
from modules.settings import BaseSettings, Field

from .playhead_offset import PlayheadOffset

# Zero-crossing guard: only treat an offset sign change as a hit when both samples are within
# a quarter-turn of the playhead, so the ±π wrap (opposite side of the ring) never counts.
_HALF_PI: float = math.pi / 2.0


class PlayheadStability(NormalizedSingleValue):
    """Per-person pose consistency across playhead sweeps, in ``[0, 1]``.

    ``0.0`` with no banked sweeps; ``1.0`` only when the live pose matches a full ring of
    banked sweeps. ``0.0`` with score ``0.0`` before the first hit or after an azimuth reset.
    """


class PlayheadStabilityExtractorSettings(BaseSettings):
    """Configuration for ``PlayheadStabilityExtractor``."""
    max_samples:    Field[int]   = Field(4, min=1, max=8, access=Field.INIT, description="Sweeps per person (live pose compared against the previous max_samples − 1)")
    angle_scale:    Field[float] = Field(0.8, min=0.1, max=2.0, step=0.05, description="Angle similarity scale (rad)")
    azimuth_reset:  Field[float] = Field(0.35, min=0.0, max=math.pi, step=0.01, description="Azimuth drift from history start that resets the buffer (rad)")
    override:       Field[bool]  = Field(False, description="Force a fixed stability value instead of the computed one", newline=True)
    override_value: Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, description="Stability value used while override is enabled")


def _pose_similarity(current: Angles, older: Angles, scale: float) -> float:
    """Confidence-weighted harmonic-mean Gaussian similarity in ``[0, 1]`` between two poses.

    Reuses ``Angles.subtract`` for the wrapped per-joint difference and conservative (min)
    confidence — the same kernel ``WindowSimilarity`` uses. The harmonic mean is strict: a
    single dissimilar joint pulls the whole score down.
    """
    diff = current.subtract(older)
    valid = diff.scores > 0.0
    if not valid.any():
        return 0.0
    sim = np.exp(-np.square(diff.values[valid] / scale))   # per-joint Gaussian, (0, 1]
    weights = diff.scores[valid]
    return float(weights.sum() / (weights / sim).sum())     # weighted harmonic mean


class PlayheadStabilityExtractor(FilterNode):
    """Samples the pose at each playhead hit and stamps ``PlayheadStability``.

    Holds per-track state (one instance per ``FilterPipeline``); ``reset()`` is called
    automatically by ``FilterTracker`` when the track is lost. Reads ``PlayheadOffset``
    (stamped upstream in the same ``filters_lerp`` pipeline), ``Azimuth`` and ``Angles``;
    writes ``PlayheadStability``. The value tracks the live pose against the banked sweeps
    every frame; the ring advances by one banked sweep at each hit.
    """

    def __init__(self, settings: PlayheadStabilityExtractorSettings | None = None) -> None:
        self._settings = settings if settings is not None else PlayheadStabilityExtractorSettings()
        self.reset()

    def reset(self) -> None:
        # Banked poses from past playhead crossings; the live pose is scored against them.
        self._samples: deque[Angles] = deque(maxlen=self._settings.max_samples - 1)
        self._anchor_az:  float = float("nan")   # azimuth where the current history began
        self._prev_offset: float = float("nan")  # last frame's PlayheadOffset, for crossing detection

    def process(self, pose: Frame) -> Frame:
        offset:  float = pose[PlayheadOffset].value
        azimuth: float = pose[Azimuth].value

        # Azimuth drift (absolute from where the history began) → drop the history.
        if not math.isnan(self._anchor_az) and not math.isnan(azimuth):
            drift = abs(math.atan2(math.sin(azimuth - self._anchor_az), math.cos(azimuth - self._anchor_az)))
            if drift > self._settings.azimuth_reset:
                self.reset()

        # Hit = PlayheadOffset zero-crossing near 0 (the ±π wrap is excluded by the guard) →
        # bank the current pose as a completed sweep. NaN offset (playhead unlocked / no
        # azimuth) fails the comparison, so no hit fires.
        if (self._prev_offset * offset < 0.0
                and abs(self._prev_offset) < _HALF_PI and abs(offset) < _HALF_PI):
            self._on_hit(pose[Angles], azimuth)
        self._prev_offset = offset

        if self._settings.override:
            return replace(pose, {PlayheadStability: PlayheadStability.from_value(self._settings.override_value, 1.0)})

        # Score the *live* pose against the banked sweeps every frame, so the value tracks
        # what the person is doing now instead of lagging behind the last sampled sweep.
        # No banked sweeps yet (before first hit / after a reset) → value 0.0, score 0.0.
        if not self._samples:
            return replace(pose, {PlayheadStability: PlayheadStability.from_value(0.0, 0.0)})
        stability = self._compute_stability(pose[Angles])
        return replace(pose, {PlayheadStability: PlayheadStability.from_value(stability, 1.0)})

    def _on_hit(self, angles: Angles, azimuth: float) -> None:
        if not self._samples:
            self._anchor_az = azimuth        # anchor where a fresh history begins
        self._samples.append(angles)         # bank this sweep; ring caps at max_samples − 1

    def _compute_stability(self, current: Angles) -> float:
        scale = self._settings.angle_scale
        total = sum(_pose_similarity(current, older, scale) for older in self._samples)
        # Σ sim / (max_samples − 1): folds match quality × buffer fill; 1.0 only when the
        # ring is full and every banked sweep matches the live pose.
        return total / (self._settings.max_samples - 1)
