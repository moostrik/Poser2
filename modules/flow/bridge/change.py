"""The bandpass's four changes."""

from enum import IntEnum


class Change(IntEnum):
    """The four changes: a rise of fast - slow along the motion that its region shares (START) or not (BOW,
    a limb arriving), a fall that its region shares (STOP) or not (WAKE, a limb leaving)."""
    START = 0
    BOW = 1
    STOP = 2
    WAKE = 3

    @property
    def sign(self) -> float:
        """+1 for a rise, -1 for a fall."""
        return 1.0 if self in (Change.START, Change.BOW) else -1.0

    @property
    def edge(self) -> bool:
        """Whether it is the part its region does not share."""
        return self in (Change.BOW, Change.WAKE)
