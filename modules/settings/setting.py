"""Setting descriptor with type coercion, callbacks, and thread safety."""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Generic, TypeVar, overload, Any

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Setting(Generic[T]):
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
        type_: type[T],
        default: T,
        *,
        min: T | None = None,
        max: T | None = None,
        step: T | None = None,
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
        self.name = ""

    def __set_name__(self, owner, name):
        self.name = name

    # -- Descriptor protocol -------------------------------------------------

    @overload
    def __get__(self, obj: None, objtype: type) -> Setting[T]: ...
    @overload
    def __get__(self, obj: Any, objtype: type) -> T: ...
    def __get__(self, obj: Any | None, objtype: type | None = None) -> Setting[T] | T:
        if obj is None:
            return self  # class-level access returns the descriptor itself
        return obj._values[self.name]

    def __set__(self, obj: Any, value: T) -> None:
        self.set(obj, value)

    # -- Public set (respects init_only) ------------------------------------

    def set(self, obj, value):
        """Set value. Only enforces init_only after initialization."""
        if self.init_only and obj._initialized:
            raise AttributeError(
                f"Setting '{self.name}' can only be set during initialization"
            )
        value = self._coerce(value)
        lock = obj._locks[self.name]
        callbacks_to_fire = []

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

    def _coerce(self, value):
        """Coerce *value* to self.type_, or raise TypeError."""
        # Reject bool when int or float is expected (bool is a subclass of int)
        if self.type_ in (int, float) and isinstance(value, bool):
            raise TypeError(
                f"Setting '{self.name}' expects {self.type_.__name__}, got bool: {value!r}"
            )
        if isinstance(value, self.type_):
            return value
        # Promote int → float (lossless)
        if self.type_ is float and isinstance(value, int):
            return float(value)
        # Enum: reconstruct from stored .value
        if isinstance(self.type_, type) and issubclass(self.type_, Enum):
            # JSON round-trips tuples as lists — restore tuple for hashable lookup
            if isinstance(value, list):
                value = tuple(value)
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

    def add_callback(self, obj, callback):
        with obj._locks[self.name]:
            if callback not in obj._callbacks[self.name]:
                obj._callbacks[self.name].append(callback)

    def remove_callback(self, obj, callback):
        with obj._locks[self.name]:
            try:
                obj._callbacks[self.name].remove(callback)
            except ValueError:
                pass

    # -- Serialization -------------------------------------------------------

    def to_json_value(self, obj):
        """Return a JSON-serializable representation of the current value."""
        value = obj._values[self.name]
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if isinstance(value, Enum):
            return value.value
        return value

    def from_json_value(self, obj, raw):
        """Restore value from JSON via set().

        Will raise AttributeError on init_only fields after initialization.
        Callers (e.g. update_from_dict) are responsible for skipping those.
        """
        self.set(obj, raw)

    # -- Repr ----------------------------------------------------------------

    def __repr__(self):
        parts = [f"{self.type_.__name__}", f"default={self.default!r}"]
        if self.readonly:
            parts.append("readonly=True")
        if self.init_only:
            parts.append("init_only=True")
        if not self.visible:
            parts.append("visible=False")
        return f"Setting({', '.join(parts)})"
