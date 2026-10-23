"""HitSync — the show's sync condition on the hits: how many of the last hits struck alike poses.

The playhead crosses each player once per round and plays that pose's sound; that crossing is the hit
(``PlayheadCrossing``, the one rule the flash, the instrument and the show share). This component records the
hit player's LERP pose at that tick and compares the most recent hits with the pipeline's own posture module
on its own settings (``posture_distance`` / ``posture_similarity``, ``pose.similarity.posture``). A run of
hits is **in sync** when two sentences hold: every pair of its poses is fully alike — the similarity reads 1,
its plateau, the postures within ``angle_tolerance`` — and, while the neutral weight is enabled, every one of
them is fully out of neutral (``ArmDeviation`` 1, at or beyond the arm extractor's ``max_degrees``), so a
neutral hit — the glass ping — is alike to nothing. The streak is the longest such run of the most recent
hits spanning less than one round (bar). A hit that does not match becomes the newest hit and the count
restarts from it at once. The tolerance, in degrees, is the sync's only knob, and the sync is the test of it:
too strict and the show never spins up.

Published on the board per light tick (``HitStreak``): the state machine spins the show up at ``min_players``
alike hits in a row, and anything else may read the tension building. The readout is in degrees: the largest
distance between the hits in the streak. The dummy's hits count like a player's when it is enabled. Ticked by
the conductor on the light thread, before the state machine.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from modules.board import HitStreak
from modules.pose import features
from modules.pose.frame import Frame
from modules.pose.analytics import PostureSimilaritySettings, posture_distance, posture_similarity
from modules.settings import BaseSettings, Field, Widget

from .neutral_weight import NeutralWeightSettings
from .playhead_offset import PlayheadCrossing, PlayheadOffset, playhead_step

if TYPE_CHECKING:
    from ..board import Board
    from ..light import LightSettings

_KEPT_HITS = 16          # more than a round of the largest room
_FULL = 1.0 - 1e-6       # "reads 1" for a float: the similarity's plateau, the deviation's top


class HitSyncSettings(BaseSettings):
    """The sync condition's telemetry; its one tunable is the posture similarity's ``angle_tolerance``."""
    hits:     Field[int]   = Field(0, access=Field.READ, pinned=True, description="Hits in a row that struck alike poses, this round")
    distance: Field[float] = Field(0.0, min=0.0, max=180.0, widget=Widget.number, access=Field.READ, description="Largest distance (°) between the hits in the streak; 0 with fewer than two")


@dataclass(frozen=True, slots=True)
class _Hit:
    """One hit: when (in bars), whose, and the pose as it was."""
    bar:  float
    id:   int
    pose: Frame


class HitSync:
    """Records the hits' poses and publishes the streak; see the module docstring."""

    def __init__(self, config: HitSyncSettings, posture: PostureSimilaritySettings, neutral: NeutralWeightSettings,
                 light: LightSettings, board: Board, pose_stage: int) -> None:
        self._config = config
        self._posture = posture
        self._neutral = neutral
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
            self._hits.append(_Hit(signals.bars, id, frames[id]))
        expired = False
        while self._hits and signals.bars - self._hits[0].bar >= 1.0:
            self._hits.popleft()
            expired = True
        if ids or expired:
            self._streak = self._compute_streak()

        hits, distance = self._streak
        self._config.hits = hits
        self._config.distance = distance
        self._board.set_hit_streak(HitStreak(hit=bool(ids), hits=hits, distance=distance))

    def _compute_streak(self) -> tuple[int, float]:
        """The longest run of the most recent hits, spanning less than one bar, that is in sync (the module
        docstring's two sentences); ``(k, the largest pair distance in it)``. One hit alone is a run of 1
        with distance 0; no hit is ``(0, 0.0)``."""
        hits = list(self._hits)
        n = len(hits)
        if n == 0:
            return 0, 0.0
        distances = self._pair_distances(hits)
        posing = [self._is_posing(hit.pose) for hit in hits]
        newest = hits[-1].bar
        for size in range(n, 1, -1):
            first = n - size
            if newest - hits[first].bar >= 1.0:
                continue                                        # a hit from a round ago is not in this run
            if not all(posing[first:]):
                continue
            run = distances[first:, first:][np.triu_indices(size, k=1)]
            if np.isnan(run).any():
                continue
            if all(posture_similarity(float(d), self._posture) >= _FULL for d in run):
                return size, float(run.max())
        return 1, 0.0

    def _is_posing(self, pose: Frame) -> bool:
        """Fully out of neutral, or the neutral weight is off; arms not seen are neutral."""
        if not self._neutral.enabled:
            return True
        deviation = pose[features.ArmDeviation].value
        return not math.isnan(deviation) and deviation >= _FULL

    def _pair_distances(self, hits: list[_Hit]) -> np.ndarray:
        """The pairwise posture distance of the hits' poses, degrees; NaN where two share no selected joint."""
        n = len(hits)
        pair = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                distance, _ = posture_distance(hits[i].pose[features.Angles], hits[j].pose[features.Angles], self._posture)
                pair[i, j] = pair[j, i] = distance
        return pair
