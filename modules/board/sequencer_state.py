from __future__ import annotations

from threading import Lock
from typing import Protocol

from modules.session import SequencerState


class HasSequencerState(Protocol):
    """Sequencer state access."""
    def get_sequencer_state(self) -> SequencerState: ...
    def set_sequencer_state(self, state: SequencerState) -> None: ...


class SequencerStateStoreMixin:
    """Thread-safe sequencer state storage."""

    def __init__(self) -> None:
        self._sequencer_state_lock = Lock()
        self._sequencer_state: SequencerState = SequencerState(
            stage=0, stage_progress=0.0, progress=0.0, elapsed=0.0, active=False,
        )

    def get_sequencer_state(self) -> SequencerState:
        with self._sequencer_state_lock:
            return self._sequencer_state

    def set_sequencer_state(self, state: SequencerState) -> None:
        with self._sequencer_state_lock:
            self._sequencer_state = state
