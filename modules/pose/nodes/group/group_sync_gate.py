"""GroupSyncGate — the group's similarity level and the gate on it, written to the group record.

A group step: the frame's ``Similarity`` rows as one matrix (``analytics.similarity_matrix``), the group's level
by the mode (``analytics.group_similarity_level``: the least alike pair, or the best level with the one who drags
the group down most left out), and the gate with hysteresis: it opens when the level reads ``on_at`` and closes
only when it falls below ``off_at``, so a group hovering at one level never swaps it. A level that cannot be
read (fewer than two poses, a pair with no similarity) closes the gate. Writes ``GroupSimilarity`` and
``GroupSync`` to the record and the panel's readouts; the poses stay as they are. Synchronous, on the frame's
thread.
"""

from __future__ import annotations

import math

from modules.settings import BaseSettings, Field, Widget

from ...analytics import GroupSimilarityMode, group_similarity_level, similarity_matrix
from ...features import GroupSimilarity, GroupSync
from ...frame import FrameDict, GroupRecord, replace_group
from ..nodes import GroupNode


class GroupSyncSettings(BaseSettings):
    """The gate's mode and band, and its readouts."""
    mode:   Field[GroupSimilarityMode] = Field(GroupSimilarityMode.ALL, description="The group's level: the least alike pair of all, or with the one who drags it down most left out (3 or more)")
    on_at:  Field[float] = Field(0.95, min=0.0, max=1.0, step=0.01, description="Group sync comes on when the level reads at least this")
    off_at: Field[float] = Field(0.90, min=0.0, max=1.0, step=0.01, description="Group sync goes off when the level falls below this")
    level:  Field[float] = Field(0.0, min=0.0, max=1.0, widget=Widget.slider, access=Field.READ, description="The group's similarity level; 0 when it cannot be read")
    on:     Field[bool]  = Field(False, access=Field.READ, pinned=True, label="Group Sync", description="The gate is open: the group is in sync")


class GroupSyncGate(GroupNode):
    """Writes the group's level and the gate on it; see the module docstring."""

    def __init__(self, settings: GroupSyncSettings | None = None) -> None:
        self.settings = settings if settings is not None else GroupSyncSettings()
        self._on = False

    def record(self, frames: FrameDict, group: GroupRecord) -> GroupRecord:
        P = self.settings
        _, matrix = similarity_matrix(frames)
        level = group_similarity_level(matrix, GroupSimilarityMode(int(P.mode)))
        if math.isnan(level):
            self._on = False
        else:
            self._on = level >= (min(P.off_at, P.on_at) if self._on else P.on_at)
        P.level = 0.0 if math.isnan(level) else level
        P.on = self._on
        return replace_group(group, {
            GroupSimilarity: GroupSimilarity.create_dummy() if math.isnan(level) else GroupSimilarity.from_value(level),
            GroupSync: GroupSync.from_value(1.0 if self._on else 0.0),
        })

    def reset(self) -> None:
        """The gate closes; it needs ``on_at`` again."""
        self._on = False
