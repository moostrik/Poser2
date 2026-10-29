"""SequencerState — the show-state snapshot a show's sequencer or state machine publishes to the
board and the OSC sound; the wire format they share."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SequencerState:
    """Immutable snapshot of the show's current state."""
    stage: int
    stage_progress: float
    progress: float
    elapsed: float
    active: bool
