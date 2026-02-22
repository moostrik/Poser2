"""Setting descriptor with type coercion, callbacks, and thread safety."""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class Setting:
    """Descriptor-based setting field with type coercion, callbacks, and thread safety.

    Declare as a class attribute on a BaseSettings subclass::

        exposure = Setting(int, 1000, min=100, max=10000)

    The descriptor handles get/set, type enforcement, init-only guards,
    callbacks, and JSON serialization.

    Custom types used as Setting values should implement ``__eq__`` so that
    change detection (skip callback when value unchanged) works correctly.
    """

    def __init__(
        self,
        type_: type,
        default: Any,
        *,
        min: float | None = None,
        max: float | None = None,
        step: float | None = None,
        description: str = "",
        readonly: bool = False,
        init_only: bool = False,
        visible: bool = True,
    ) -> None:
        self.type_ = type_
        self.default = default
        self.min = min
        self.max = max
        self.step = step
        self.description = description
        self.readonly = readonly
        self.init_only = init_only
        self.visible = visible
        # Set by __set_name__
        self.name: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    # -- Descriptor protocol -------------------------------------------------

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self  # class-level access returns the descriptor itself
        return obj._values[self.name]

    def __set__(self, obj: Any, value: Any) -> None:
        self.set(obj, value)

    # -- Public set (respects init_only) ------------------------------------

    def set(self, obj: Any, value: Any) -> None:
        """Set value. Only enforces init_only after initialization."""
        if self.init_only and obj._initialized:
            raise AttributeError(
                f"Setting '{self.name}' can only be set during initialization"
            )
        value = self._coerce(value)
        lock = obj._locks[self.name]
        callbacks_to_fire: list[Callable] = []

        with lock:
            if obj._values[self.name] != value:
                obj._values[self.name] = value
                callbacks_to_fire = list(obj._callbacks[self.name])

        # Fire callbacks outside the lock
        for cb in callbacks_to_fire:
            try:
                cb(value)
            except Exception:
                logger.warning(
                    "Callback %r for setting '%s' raised an exception",
                    cb, self.name, exc_info=True,
                )

    # -- Type coercion -------------------------------------------------------

    def _coerce(self, value: Any) -> Any:
        """Coerce *value* to self.type_, or raise TypeError."""
        # Reject bool when int is expected (bool is a subclass of int)
        if self.type_ is int and isinstance(value, bool):
            raise TypeError(
                f"Setting '{self.name}' expects int, got bool: {value!r}"
            )
        if isinstance(value, self.type_):
            return value
        # Enum: reconstruct from stored .value
        if isinstance(self.type_, type) and issubclass(self.type_, Enum):
            try:
                return self.type_(value)
            except (ValueError, KeyError):
                raise TypeError(
                    f"Cannot construct {self.type_.__name__} from {value!r}"
                )
        if isinstance(value, (tuple, list)):
            try:
                return self.type_(*value)
            except Exception:
                raise TypeError(
                    f"Cannot construct {self.type_.__name__} from {type(value).__name__}: {value!r}"
                )
        if isinstance(value, dict) and hasattr(self.type_, "from_dict"):
            try:
                return self.type_.from_dict(value)  # type: ignore[union-attr]
            except Exception:
                raise TypeError(
                    f"Cannot construct {self.type_.__name__} from dict: {value!r}"
                )
        raise TypeError(
            f"Setting '{self.name}' expects {self.type_.__name__}, "
            f"got {type(value).__name__}: {value!r}"
        )

    # -- Callbacks -----------------------------------------------------------

    def add_callback(self, obj: Any, callback: Callable) -> None:
        with obj._locks[self.name]:
            if callback not in obj._callbacks[self.name]:
                obj._callbacks[self.name].append(callback)

    def remove_callback(self, obj: Any, callback: Callable) -> None:
        with obj._locks[self.name]:
            try:
                obj._callbacks[self.name].remove(callback)
            except ValueError:
                pass

    # -- Serialization -------------------------------------------------------

    def to_json_value(self, obj: Any) -> Any:
        """Return a JSON-serializable representation of the current value."""
        value = obj._values[self.name]
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if isinstance(value, Enum):
            return value.value
        return value

    def from_json_value(self, obj: Any, raw: Any) -> None:
        """Restore value from JSON via set().

        Will raise AttributeError on init_only fields after initialization.
        Callers (e.g. update_from_dict) are responsible for skipping those.
        """
        self.set(obj, raw)

    # -- Repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"{self.type_.__name__}", f"default={self.default!r}"]
        if self.readonly:
            parts.append("readonly=True")
        if self.init_only:
            parts.append("init_only=True")
        if not self.visible:
            parts.append("visible=False")
        return f"Setting({', '.join(parts)})"
