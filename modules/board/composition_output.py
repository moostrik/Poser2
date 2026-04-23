from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from apps.white_space.composition.output import CompositionOutput, CompositionDebug


class HasCompositionOutput(Protocol):
    """Composition pipeline output access."""
    def get_composition_output(self) -> CompositionOutput | None: ...
    def set_composition_output(self, output: CompositionOutput) -> None: ...


class CompositionOutputStoreMixin:
    """Thread-safe composition output storage."""

    def __init__(self) -> None:
        self._composition_output_lock = Lock()
        self._composition_output: CompositionOutput | None = None

    def get_composition_output(self) -> CompositionOutput | None:
        with self._composition_output_lock:
            return self._composition_output

    def set_composition_output(self, output: CompositionOutput) -> None:
        with self._composition_output_lock:
            self._composition_output = output


class HasCompositionDebug(Protocol):
    """Composition debug channels access."""
    def get_composition_debug(self) -> CompositionDebug | None: ...
    def set_composition_debug(self, debug: CompositionDebug) -> None: ...


class CompositionDebugStoreMixin:
    """Thread-safe composition debug storage."""

    def __init__(self) -> None:
        self._composition_debug_lock = Lock()
        self._composition_debug: CompositionDebug | None = None

    def get_composition_debug(self) -> CompositionDebug | None:
        with self._composition_debug_lock:
            return self._composition_debug

    def set_composition_debug(self, debug: CompositionDebug) -> None:
        with self._composition_debug_lock:
            self._composition_debug = debug
