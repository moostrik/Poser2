"""PlayheadStability — a White Space-native pose feature: how consistently a person holds
their pose across successive playhead sweeps. It measures the *pose's* consistency; the
playhead is only the sampling trigger.

Each time the rotating playhead crosses a person (a ``PlayheadOffset`` zero-crossing) the
current pose (``Angles``) is sampled into a short per-person ring buffer. Stability is the
summed similarity of the current sample to the older ones, normalised by buffer capacity:
it is ``0.0`` with a single sample and only reaches ``1.0`` with a full buffer of matching
poses. The history (and the value) reset when the person's azimuth drifts too far from
where the history began.

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

    ``0.0`` with a single sample; ``1.0`` only with a full sample buffer of matching poses.
    ``0.0`` with score ``0.0`` before the first hit or after an azimuth reset.
    """


class PlayheadStabilityExtractorSettings(BaseSettings):
    """Configuration for ``PlayheadStabilityExtractor``."""
    max_samples:   Field[int]   = Field(4, min=1, max=8, access=Field.INIT, description="Pose samples retained per person")
    angle_scale:   Field[float] = Field(0.8, min=0.1, max=2.0, step=0.05, description="Angle similarity scale (rad)")
    azimuth_reset: Field[float] = Field(0.35, min=0.0, max=math.pi, step=0.01, description="Azimuth drift from history start that resets the buffer (rad)")


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
    writes ``PlayheadStability``. The value holds between hits (a step per sweep).
    """

    def __init__(self, settings: PlayheadStabilityExtractorSettings | None = None) -> None:
        self._settings = settings if settings is not None else PlayheadStabilityExtractorSettings()
        self.reset()

    def reset(self) -> None:
        self._samples: deque[Angles] = deque(maxlen=self._settings.max_samples)
        self._anchor_az:  float = float("nan")   # azimuth where the current history began
        self._prev_offset: float = float("nan")  # last frame's PlayheadOffset, for crossing detection
        self._stability:  float = 0.0            # last computed value, held between hits (0.0 until first hit)

    def process(self, pose: Frame) -> Frame:
        offset:  float = pose[PlayheadOffset].value
        azimuth: float = pose[Azimuth].value

        # Azimuth drift (absolute from where the history began) → drop the history.
        if not math.isnan(self._anchor_az) and not math.isnan(azimuth):
            drift = abs(math.atan2(math.sin(azimuth - self._anchor_az), math.cos(azimuth - self._anchor_az)))
            if drift > self._settings.azimuth_reset:
                self.reset()

        # Hit = PlayheadOffset zero-crossing near 0 (the ±π wrap is excluded by the guard).
        # NaN offset (playhead unlocked / no azimuth) fails the comparison, so no hit fires.
        if (self._prev_offset * offset < 0.0
                and abs(self._prev_offset) < _HALF_PI and abs(offset) < _HALF_PI):
            self._on_hit(pose[Angles], azimuth)
        self._prev_offset = offset

        # No samples yet (before first hit / after a reset) → value 0.0 with score 0.0.
        score = 1.0 if self._samples else 0.0
        return replace(pose, {PlayheadStability: PlayheadStability.from_value(self._stability, score)})

    def _on_hit(self, angles: Angles, azimuth: float) -> None:
        if not self._samples:
            self._anchor_az = azimuth        # anchor where a fresh history begins
        self._samples.append(angles)         # newest = current; ring caps at max_samples
        self._stability = self._compute_stability()

    def _compute_stability(self) -> float:
        if len(self._samples) < 2:
            return 0.0                        # one sample → 0.0
        current = self._samples[-1]
        scale = self._settings.angle_scale
        total = sum(_pose_similarity(current, older, scale) for older in tuple(self._samples)[:-1])
        # Σ sim / (capacity − 1): folds match quality × buffer fill; 1.0 only when the buffer
        # is full and every older sample matches the current one.
        return total / (self._settings.max_samples - 1)
