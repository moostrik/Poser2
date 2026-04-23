from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from apps.white_space.light.LightOutput import LightOutput, LightDebug


class HasLightOutput(Protocol):
    """LED light pipeline output access."""
    def get_light_output(self) -> LightOutput | None: ...
    def set_light_output(self, output: LightOutput) -> None: ...


class LightOutputStoreMixin:
    """Thread-safe LED light output storage."""

    def __init__(self) -> None:
        self._light_output_lock = Lock()
        self._light_output: LightOutput | None = None

    def get_light_output(self) -> LightOutput | None:
        with self._light_output_lock:
            return self._light_output

    def set_light_output(self, output: LightOutput) -> None:
        with self._light_output_lock:
            self._light_output = output


class HasLightDebug(Protocol):
    """LED light compositor debug channels access."""
    def get_light_debug(self) -> LightDebug | None: ...
    def set_light_debug(self, debug: LightDebug) -> None: ...


class LightDebugStoreMixin:
    """Thread-safe LED light debug storage."""

    def __init__(self) -> None:
        self._light_debug_lock = Lock()
        self._light_debug: LightDebug | None = None

    def get_light_debug(self) -> LightDebug | None:
        with self._light_debug_lock:
            return self._light_debug

    def set_light_debug(self, debug: LightDebug) -> None:
        with self._light_debug_lock:
            self._light_debug = debug
