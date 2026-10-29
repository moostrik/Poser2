from __future__ import annotations

from .base import NormalizedSingleValue


class GroupSync(NormalizedSingleValue):
    """The group in sync: 1 while the gate on the group similarity is open, 0 while it is closed. A group record
    feature, written by ``GroupSyncGate``; ``on`` reads it as the gate, closed when absent."""

    @property
    def on(self) -> bool:
        """The gate is open."""
        return self.value >= 0.5

    @classmethod
    def range(cls) -> tuple[float, float]:
        return (0.0, 1.0)
