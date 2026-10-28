"""Posture similarity — how far apart two postures are, and how alike that makes them.

Two answers, the second made from the first:

- **Distance**, in degrees (0 identical, 180 opposite): per selected joint the wrapped angle difference, and
  over the joints both poses have a soft maximum — the worst joint decides, but one joint is forgiven when
  the rest match. ``forgiveness`` says by how much: a lone joint at ``forgiveness × D`` with the others exact
  reads ``D``. It is a power mean whose exponent follows from that sentence, ``p = ln n / ln forgiveness``
  for the ``n`` joints compared, so the sentence holds at any coverage; below 1.1 the extra room is under
  what a pose estimate resolves, and the distance is the plain maximum. Degrees, not tolerance units: the
  same pair reads the same number whatever the tolerance is set to, so the tolerance can be tuned while
  watching it.
- **Similarity**, 0 to 1, with ``angle_tolerance`` as the slack at both ends: 1 within the tolerance of
  identical, 0 within the tolerance of opposite, a straight line between. At 20°: 1 up to 20°, 0.5 at 90°,
  0 from 160°. Exactly 1.0 on the plateau, so "fully alike" is the score reading 1.

A joint missing on either side is not compared; the coverage — the fraction of the selected joints that were
— is returned beside the distance and becomes the ``Similarity`` score. How a consumer shapes the similarity
further (a threshold, a curve) is the consumer's.

``PostureSimilarity`` computes the pairwise similarity of the current poses synchronously, on the caller's
thread; ``posture_distance`` and ``posture_similarity`` serve anything comparing two poses on their own.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from modules.settings import BaseSettings, Field, Group
from ..features import Angles, Similarity
from ..frame import FrameDict
from .joint_select import JointSelectSettings, joint_mask
from .window_similarity import SimilarityResult

_NO_FORGIVENESS = 1.1      # below this the distance is the plain maximum (see the module docstring)


class PostureSimilaritySettings(BaseSettings):
    """The definition of alike: the slack, the forgiveness for one odd joint, and the joints compared."""
    max_poses:       Field[int]   = Field(3, min=1, max=16, access=Field.INIT, description="Maximum number of tracked poses")
    angle_tolerance: Field[float] = Field(20.0, min=1.0, max=89.0, step=1.0, description="Slack (°): postures this close to identical are fully alike (1), this close to opposite fully different (0)")
    forgiveness:     Field[float] = Field(1.4, min=1.0, max=3.0, step=0.1, description="One joint may exceed the tolerance by this factor when the rest match: 1 none, 1.4 the default, 2 twice the tolerance")
    joints:          Group[JointSelectSettings] = Group(JointSelectSettings)


def _soft_max(distances: np.ndarray, forgiveness: float) -> np.ndarray:
    """The soft maximum over the last axis of joint distances (degrees, NaN where not compared); NaN where
    no joint was. Computed relative to the largest distance, so no exponent can overflow."""
    valid = ~np.isnan(distances)
    n = valid.sum(axis=-1)
    filled = np.where(valid, distances, 0.0).astype(np.float64)
    largest = filled.max(axis=-1)
    out = np.where(n > 0, largest, np.nan)
    if forgiveness < _NO_FORGIVENESS:
        return out
    soft = (n > 1) & (largest > 0.0)
    if not np.any(soft):
        return out
    p = np.log(np.where(soft, n, 2)) / math.log(forgiveness)
    ratios = filled / np.where(largest > 0.0, largest, 1.0)[..., None]
    powered = np.where(valid, ratios ** p[..., None], 0.0)
    mean = powered.sum(axis=-1) / np.maximum(n, 1)
    return np.where(soft, largest * mean ** (1.0 / p), out)


def joint_distances(a: Angles, b: Angles, mask: np.ndarray | None = None) -> np.ndarray:
    """Per joint how far apart the two poses are, degrees ``(F,)``: the wrapped difference, NaN where a joint
    is missing on either side (score 0) or not selected."""
    diff = a.subtract(b)
    distances = np.degrees(np.abs(diff.values)).astype(np.float64)
    compared = (diff.scores > 0.0) & ~np.isnan(distances)
    if mask is not None:
        compared &= mask
    return np.where(compared, distances, np.nan)


def posture_distance(a: Angles, b: Angles, config: PostureSimilaritySettings) -> tuple[float, float]:
    """``(distance in degrees, coverage)`` of two poses; ``(NaN, 0.0)`` when they share no selected joint."""
    mask = joint_mask(config.joints)
    distances = joint_distances(a, b, mask)
    selected = int(mask.sum())
    compared = int(np.sum(~np.isnan(distances)))
    if selected == 0 or compared == 0:
        return math.nan, 0.0
    return float(_soft_max(distances, float(config.forgiveness))), compared / selected


def _score(distances: np.ndarray, tolerance: float) -> np.ndarray:
    span = 180.0 - 2.0 * tolerance
    if span <= 0.0:
        return np.where(np.isnan(distances), np.nan, (distances <= tolerance).astype(np.float64))
    return np.clip((180.0 - tolerance - distances) / span, 0.0, 1.0)      # NaN stays NaN


def posture_similarity(distance: float, config: PostureSimilaritySettings) -> float:
    """The similarity of a distance: exactly 1.0 up to ``angle_tolerance``, 0.0 from ``180 − tolerance``,
    linear between; NaN for NaN."""
    return float(_score(np.asarray(distance, dtype=np.float64), float(config.angle_tolerance)))


class PostureSimilarity:
    """Pairwise posture similarity of the current poses: ``Similarity`` rows (0..1, score = coverage), indexed
    by pose id, emitted as a ``SimilarityResult`` with no leader scores. Every present pose gets a row, so a
    pose alone reads NaN toward everyone. Synchronous, on the caller's thread."""

    def __init__(self, config: PostureSimilaritySettings | None = None) -> None:
        self._config = config if config is not None else PostureSimilaritySettings()
        self._callbacks: list[Callable[[SimilarityResult], None]] = []

    def add_similarity_callback(self, callback: Callable[[SimilarityResult], None]) -> None:
        self._callbacks.append(callback)

    def process(self, frames: FrameDict) -> None:
        """Compute the rows for ``frames`` and emit them."""
        result = SimilarityResult(self._rows(frames), {})
        for callback in self._callbacks:
            callback(result)

    def _rows(self, frames: FrameDict) -> dict[int, Similarity]:
        length = Similarity.length()
        ids = [id for id in frames if id < length]
        rows: dict[int, Similarity] = {}
        if not ids:
            return rows

        mask = joint_mask(self._config.joints)
        selected = int(mask.sum())
        angles = [frames[id][Angles] for id in ids]
        values = np.stack([a.values for a in angles]).astype(np.float64)           # (N, F)
        seen = np.stack([a.scores > 0.0 for a in angles])                          # (N, F)
        diff = values[:, None, :] - values[None, :, :]
        distances = np.degrees(np.abs(np.arctan2(np.sin(diff), np.cos(diff))))     # wrapped, (N, N, F)
        compared = seen[:, None, :] & seen[None, :, :] & ~np.isnan(distances) & mask[None, None, :]
        distances = np.where(compared, distances, np.nan)

        pair_distance = _soft_max(distances, float(self._config.forgiveness))      # (N, N)
        similarity = _score(pair_distance, float(self._config.angle_tolerance))
        coverage = compared.sum(axis=-1) / selected if selected > 0 else np.zeros_like(pair_distance)

        for i, id in enumerate(ids):
            row_values = np.full(length, np.nan, dtype=np.float32)
            row_scores = np.zeros(length, dtype=np.float32)
            for j, other in enumerate(ids):
                if i == j or np.isnan(similarity[i, j]):
                    continue
                row_values[other] = similarity[i, j]
                row_scores[other] = coverage[i, j]
            rows[id] = Similarity(row_values, row_scores)
        return rows
