"""The light frame on the board: a protocol + mixin pair only White Space uses."""

from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .light.frame import LightFrame
    from .light.frame import LightFrame as CompositionDebug


class HasLightFrame(Protocol):
    """Light frame access."""
    def get_light_frame(self) -> LightFrame | None: ...
    def set_light_frame(self, frame: LightFrame) -> None: ...


class LightFrameStoreMixin:
    """Thread-safe light frame storage."""

    def __init__(self) -> None:
        self._light_frame_lock = Lock()
        self._light_frame: LightFrame | None = None

    def get_light_frame(self) -> LightFrame | None:
        with self._light_frame_lock:
            return self._light_frame

    def set_light_frame(self, frame: LightFrame) -> None:
        with self._light_frame_lock:
            self._light_frame = frame


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
