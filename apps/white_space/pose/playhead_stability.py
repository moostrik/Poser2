"""PlayheadStability — a White Space-native pose feature carrying three per-person signals
about the spot a person currently occupies, sampled by the rotating light playhead. Each
value is in ``[0, 1]`` (``PlayheadElement``):

- ``Dwell``     — normalised presence: ``0`` at the 1st playhead crossing on the spot, ``1``
                  at the ``dwell_beats``-th. Ramps *continuously* between crossings from the
                  playhead's sub-beat phase.
- ``Motion``    — how much the person has moved on the spot, ``MotionTime`` accumulated since
                  spot entry, normalised by ``motion_scale``.
- ``Stability`` — pose consistency across sweeps: the summed similarity of the live pose to a
                  ring of banked sweeps. **Gated** — stays ``0`` until both ``Dwell`` and
                  ``Motion`` reach ``1``, then builds over ``stability_beats``.

A "spot" is anchored at the first playhead crossing (a ``PlayheadOffset`` zero-crossing) and
reset when the person's azimuth drifts past ``spot_radius`` from where the spot began; the
whole feature resets with it. ``MotionTime`` is the single source of truth for motion — we
snapshot it on spot entry rather than re-integrating.

Playhead is a White Space concept, so the feature, its extractor and its settings live with
the app rather than in ``modules/pose`` — like ``PlayheadOffset`` (see ``playhead_offset.py``).
The open Frame ECS still lets the feature ride on ``Frame`` (via ``replace``) without
modules depending on app code.
"""

from __future__ import annotations

import math
from collections import deque
from enum import IntEnum

import numpy as np

from modules.pose.features import Angles, Azimuth, MotionTime, NormalizedScalarFeature
from modules.pose.frame import Frame, replace
from modules.pose.nodes import FilterNode
from modules.settings import BaseSettings, Field

from .playhead_offset import PlayheadOffset

# Zero-crossing guard: only treat an offset sign change as a hit when both samples are within
# a quarter-turn of the playhead, so the ±π wrap (opposite side of the ring) never counts.
_HALF_PI: float = math.pi / 2.0
_TWO_PI:  float = 2.0 * math.pi


class PlayheadElement(IntEnum):
    """The three per-spot values carried by ``PlayheadStability``."""
    Dwell     = 0
    Motion    = 1
    Stability = 2


class PlayheadStability(NormalizedScalarFeature[PlayheadElement]):
    """Per-spot ``Dwell`` / ``Motion`` / ``Stability`` for one person, each in ``[0, 1]``.

    Absent (NaN, score ``0.0``) before the person has settled on a spot; see the module
    docstring for how each element behaves once a spot is anchored.
    """

    @classmethod
    def enum(cls) -> type[PlayheadElement]:
        return PlayheadElement


class PlayheadStabilityExtractorSettings(BaseSettings):
    """Configuration for ``PlayheadStabilityExtractor``."""
    stability_beats: Field[int]   = Field(4, min=2, max=8, access=Field.INIT, description="Beats over which stability builds (ring size)")
    dwell_beats:     Field[int]   = Field(4, min=2, max=16, description="Beats for dwell to ramp 0→1")
    angle_scale:     Field[float] = Field(0.8, min=0.1, max=2.0, step=0.05, description="Angle similarity scale (rad)")
    motion_scale:    Field[float] = Field(5.0, min=0.1, max=50.0, step=0.1, description="Accumulated on-spot motion that maps to 1.0")
    spot_radius:     Field[float] = Field(20.0, min=0.0, max=360.0, step=1.0, description="Angular radius of the spot (deg); drifting past it starts a new spot", newline=True)
    override:        Field[bool]  = Field(False, description="Force a fixed stability value instead of the computed one", newline=True)
    override_value:  Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, description="Stability value used while override is enabled")


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


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
    """Computes the per-spot ``Dwell`` / ``Motion`` / ``Stability`` and stamps ``PlayheadStability``.

    Holds per-track state (one instance per ``FilterPipeline``); ``reset()`` is called
    automatically by ``FilterTracker`` when the track is lost. Reads ``PlayheadOffset``,
    ``Azimuth``, ``MotionTime`` and ``Angles`` (all stamped upstream in the same ``filters_lerp``
    pipeline). The spot is anchored at the first crossing and reset on azimuth drift.
    """

    def __init__(self, settings: PlayheadStabilityExtractorSettings | None = None) -> None:
        self._settings = settings if settings is not None else PlayheadStabilityExtractorSettings()
        self.reset()

    def reset(self) -> None:
        # Banked poses from gated crossings; the live pose is scored against them for stability.
        self._samples: deque[Angles] = deque(maxlen=max(self._settings.stability_beats - 1, 1))
        self._anchor_az:     float = float("nan")   # azimuth where the current spot began
        self._anchor_motion: float = float("nan")   # MotionTime snapshot at spot entry
        self._spot_beats:    int   = 0              # playhead crossings since the spot began
        self._prev_offset:   float = float("nan")   # last frame's PlayheadOffset, for crossing detection

    def process(self, pose: Frame) -> Frame:
        offset:      float = pose[PlayheadOffset].value
        azimuth:     float = pose[Azimuth].value
        motion_time: float = pose[MotionTime].value

        # Azimuth drift past the spot radius → the person left the spot; start fresh.
        if not math.isnan(self._anchor_az) and not math.isnan(azimuth):
            drift = abs(math.atan2(math.sin(azimuth - self._anchor_az), math.cos(azimuth - self._anchor_az)))
            if drift > math.radians(self._settings.spot_radius):
                self.reset()

        # A hit is a PlayheadOffset zero-crossing near 0 (the ±π wrap is excluded by the guard).
        # NaN offset (playhead unlocked / no azimuth) fails the comparison, so no hit fires.
        crossing = (self._prev_offset * offset < 0.0
                    and abs(self._prev_offset) < _HALF_PI and abs(offset) < _HALF_PI)
        if crossing:
            if self._spot_beats == 0:                 # first crossing anchors a fresh spot
                self._anchor_az = azimuth
            self._spot_beats += 1
        self._prev_offset = offset

        active = self._spot_beats > 0
        dwell = self._dwell(offset) if active else 0.0
        # Motion only starts counting once dwell is full — snapshot MotionTime at that moment,
        # so the order is dwell → motion → stability rather than dwell and motion in parallel.
        if active and dwell >= 1.0 and math.isnan(self._anchor_motion):
            self._anchor_motion = motion_time
        motion = self._motion(motion_time) if active else 0.0

        # Stability is gated behind full dwell AND full motion (both monotonic per spot, so the
        # gate latches). Bank a pose only on a gated crossing → the ring holds active-phase poses.
        if crossing and active and dwell >= 1.0 and motion >= 1.0:
            self._samples.append(pose[Angles])

        if self._settings.override:
            stability, stability_score = self._settings.override_value, 1.0
        elif self._samples:
            stability, stability_score = self._compute_stability(pose[Angles]), 1.0
        else:
            stability, stability_score = 0.0, (1.0 if active else 0.0)

        presence_score = 1.0 if active else 0.0
        values = np.array([dwell, motion, stability], dtype=np.float32)
        scores = np.array([presence_score, presence_score, stability_score], dtype=np.float32)
        return replace(pose, {PlayheadStability: PlayheadStability(values=values, scores=scores)})

    def _dwell(self, offset: float) -> float:
        # Continuous beat progress: whole beats since entry + sub-beat phase from the playhead.
        # phase = fraction of the sweep since the last crossing (0 just after → 1 approaching next).
        phase = 0.0 if math.isnan(offset) else ((-offset) % _TWO_PI) / _TWO_PI
        progress = (self._spot_beats - 1) + phase
        return _clamp01(progress / max(self._settings.dwell_beats - 1, 1))

    def _motion(self, motion_time: float) -> float:
        if math.isnan(motion_time) or math.isnan(self._anchor_motion):
            return 0.0
        return _clamp01((motion_time - self._anchor_motion) / self._settings.motion_scale)

    def _compute_stability(self, current: Angles) -> float:
        scale = self._settings.angle_scale
        total = sum(_pose_similarity(current, older, scale) for older in self._samples)
        # Σ sim / (stability_beats − 1): folds match quality × ring fill; 1.0 only when the ring
        # is full and every banked sweep matches the live pose.
        return total / max(self._settings.stability_beats - 1, 1)
