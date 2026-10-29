"""Group similarity — how alike a group is, from the pairwise similarity of its poses.

``similarity_matrix`` turns a frame's ``Similarity`` rows into one symmetric matrix: a pair reads the mean of
its two directions where both read (an app that chases each row on its own makes them differ), the one that
reads else, NaN where neither does. ``group_similarity_level`` is the group's level from such a matrix, by mode:

- **ALL**: the least alike pair of everyone.
- **ALL_BUT_ONE**: the least alike pair once the one person who drags the group down most is left out, so
  three or more can be in sync while one is not: the best level any leave-one-out gives. With two there is
  nobody to leave out and it reads as ALL.

A pair that cannot be read (NaN) reads NaN into every level that needs it; with fewer than two poses there is
no pair and the level is NaN. Stateless; it knows no stages, frames' timing, board or show.
"""

from __future__ import annotations

import math
from enum import IntEnum

import numpy as np

from ..features import Similarity
from ..frame import FrameDict


class GroupSimilarityMode(IntEnum):
    ALL         = 0     # the least alike pair of everyone
    ALL_BUT_ONE = 1     # the least alike pair once the one who drags the group down most is left out


def similarity_matrix(frames: FrameDict) -> tuple[list[int], np.ndarray]:
    """The poses' ids in frame order and their pairwise similarity ``(N, N)``, symmetric, NaN on the diagonal
    and where neither direction reads; see the module docstring."""
    ids = list(frames)
    n = len(ids)
    rows = np.full((n, n), np.nan, dtype=np.float64)
    for i, id in enumerate(ids):
        values = frames[id][Similarity].values
        for j, other in enumerate(ids):
            if i != j and other < len(values):
                rows[i, j] = values[other]
    with np.errstate(invalid='ignore'):
        both = ~np.isnan(rows) & ~np.isnan(rows.T)
        matrix = np.where(both, (rows + rows.T) / 2.0, np.where(np.isnan(rows), rows.T, rows))
    return ids, matrix


def _least_alike(matrix: np.ndarray) -> float:
    """The smallest pair value of a symmetric matrix; NaN with fewer than two or a NaN pair."""
    n = len(matrix)
    if n < 2:
        return math.nan
    pairs = matrix[np.triu_indices(n, k=1)]
    return math.nan if np.isnan(pairs).any() else float(pairs.min())


def group_similarity_level(matrix: np.ndarray, mode: GroupSimilarityMode = GroupSimilarityMode.ALL) -> float:
    """The group's similarity level, 0..1, from its pairwise matrix; see the module docstring."""
    n = len(matrix)
    if mode == GroupSimilarityMode.ALL or n < 3:
        return _least_alike(matrix)
    levels = [_least_alike(np.delete(np.delete(matrix, k, axis=0), k, axis=1)) for k in range(n)]
    kept = [level for level in levels if not math.isnan(level)]
    return max(kept) if kept else math.nan
