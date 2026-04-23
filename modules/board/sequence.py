from __future__ import annotations

from threading import Lock
from typing import Protocol

from modules.session import SequencerState


class HasSequence(Protocol):
    """Sequencer state access."""
    def get_sequence(self) -> SequencerState: ...
    def set_sequence(self, state: SequencerState) -> None: ...


class SequenceStoreMixin:
    """Thread-safe sequencer state storage."""

    def __init__(self) -> None:
        self._sequence_lock = Lock()
        self._sequence: SequencerState = SequencerState(
            stage=0, stage_progress=0.0, progress=0.0, elapsed=0.0, active=False,
        )

    def get_sequence(self) -> SequencerState:
        with self._sequence_lock:
            return self._sequence

    def set_sequence(self, state: SequencerState) -> None:
        with self._sequence_lock:
            self._sequence = state
