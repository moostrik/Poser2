"""HitSync — the show's sync condition on the hits: how many of the last hits struck alike poses.

The playhead crosses each player once per round and plays that pose's sound; that crossing is the hit
(``PlayheadCrossing``, the one rule the flash, the instrument and the show share). This component records the
hit player's pose at that tick — the calibrated ``Angles`` and the ``ArmDeviation`` off their LERP frame — and
scores the most recent hits against each other with the pipeline's own posture kernel and settings
(``posture_similarity``, ``pose.similarity.window_similarity``), weighted at neutral by the less-moved of the
pair, so a neutral hit — the glass ping — is alike to nothing. The streak is the longest run of the most recent
hits, spanning less than one round (bar), that is **in sync**: each hit's harmonic mean toward the others at or
above ``threshold``. A hit that does not match becomes the newest hit and the count restarts from it at once.

Published on the board per light tick (``HitStreak``): the state machine spins the show up at ``min_players``
alike hits in a row, and anything else may read the tension building. The dummy's hits count like a player's
when it is enabled. Ticked by the conductor on the light thread, before the state machine.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from modules.board import HitStreak
from modules.pose import features
from modules.pose.analytics import WindowSimilaritySettings, posture_similarity
from modules.settings import BaseSettings, Field, Widget

from .playhead_offset import PlayheadCrossing, PlayheadOffset, playhead_step

if TYPE_CHECKING:
    from ..board import Board
    from ..light import LightSettings

_TINY = 1e-5             # the zero guard NormalizedScalarFeature uses for its harmonic mean
_KEPT_HITS = 16          # more than a round of the largest room


class HitSyncSettings(BaseSettings):
    """The sync condition on the hits: its tunable and its telemetry."""
    threshold:  Field[float] = Field(0.75, min=0.0, max=1.0, step=0.01, widget=Widget.slider, description="Two hit poses count as alike at this similarity")
    hits:       Field[int]   = Field(0, access=Field.READ, pinned=True, description="Hits in a row that struck alike poses, this round")
    similarity: Field[float] = Field(0.0, min=0.0, max=1.0, widget=Widget.number, access=Field.READ, description="Mean similarity of those hits")


@dataclass(frozen=True, slots=True)
class _Hit:
    """One hit: whose pose, when (in bars), and how far their arms were out of neutral."""
    bar:       float
    id:        int
    angles:    features.Angles
    deviation: float


class HitSync:
    """Records the hits' poses and publishes the streak; see the module docstring."""

    def __init__(self, config: HitSyncSettings, similarity: WindowSimilaritySettings, light: LightSettings,
                 board: Board, pose_stage: int) -> None:
        self._config = config
        self._similarity = similarity
        self._light = light
        self._board = board
        self._pose_stage = pose_stage
        self._crossing = PlayheadCrossing()
        self._hits: deque[_Hit] = deque(maxlen=_KEPT_HITS)
        self._streak: tuple[int, float] = (0, 0.0)

    def reset(self) -> None:
        """Forget the hits and the passes under way."""
        self._hits.clear()
        self._crossing.reset()
        self._streak = (0, 0.0)

    def update(self) -> None:
        """One light tick: record this tick's hits, drop those older than a round, publish the streak."""
        signals = self._board.get_playhead_signals()
        frames = self._board.get_frames(self._pose_stage)
        step = playhead_step(self._light.motor.beam_rpm, 1.0 / self._light.light_rate)
        offsets = {id: frame[PlayheadOffset].value for id, frame in frames.items()}
        ids = self._crossing.update(offsets, step, 1)

        for id in sorted(ids):
            frame = frames[id]
            deviation = frame[features.ArmDeviation].value
            self._hits.append(_Hit(signals.bars, id, frame[features.Angles],
                                   0.0 if math.isnan(deviation) else float(deviation)))
        expired = False
        while self._hits and signals.bars - self._hits[0].bar >= 1.0:
            self._hits.popleft()
            expired = True
        if ids or expired:
            self._streak = self._compute_streak()

        hits, similarity = self._streak
        self._config.hits = hits
        self._config.similarity = similarity
        self._board.set_hit_streak(HitStreak(hit=bool(ids), hits=hits, similarity=similarity))

    def _compute_streak(self) -> tuple[int, float]:
        """The longest run of the most recent hits, spanning less than one bar, that is in sync: each hit's
        harmonic mean of its pair values toward the others at or above ``threshold``; ``(k, mean)``. One hit
        alone is a run of 1 with similarity 0; no hit is ``(0, 0.0)``."""
        hits = list(self._hits)
        n = len(hits)
        if n == 0:
            return 0, 0.0
        pair = self._pair_values(hits)
        threshold = self._config.threshold
        newest = hits[-1].bar
        for size in range(n, 1, -1):
            members = range(n - size, n)
            if newest - hits[n - size].bar >= 1.0:
                continue                                        # a hit from a round ago is not in this run
            values: list[float] = []
            for i in members:
                s = np.array([pair[i, j] for j in members if j != i])
                if np.isnan(s).any():
                    break
                values.append(float(s.size / np.sum(1.0 / np.maximum(s, _TINY))))
            if len(values) == size and min(values) >= threshold:
                return size, float(np.mean(values))
        return 1, 0.0

    def _pair_values(self, hits: list[_Hit]) -> np.ndarray:
        """The pairwise similarity of the hits' poses, weighted by the less-moved of the pair at neutral."""
        n = len(hits)
        pair = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                value = posture_similarity(hits[i].angles, hits[j].angles, self._similarity)
                pair[i, j] = pair[j, i] = value * min(hits[i].deviation, hits[j].deviation)
        return pair
