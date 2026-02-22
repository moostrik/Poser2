"""Action descriptor — a stateless trigger with callbacks."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)


class Action:
    """A stateless trigger that fires callbacks when invoked.

    Declare as a class attribute on a BaseSettings subclass::

        reset = Action(description="Reset all values")

    The GUI renders this as a button. Calling ``action.fire(obj)``
    invokes all registered callbacks.
    """

    def __init__(
        self,
        *,
        description: str = "",
        visible: bool = True,
    ) -> None:
        self.description = description
        self.visible = visible
        self.name: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> "Action":
        if obj is None:
            return self
        return self

    def __set__(self, obj: Any, value: Any) -> None:
        raise AttributeError(
            f"Action '{self.name}' is not assignable — call fire() instead"
        )

    # -- Fire ----------------------------------------------------------------

    def fire(self, obj: Any) -> None:
        """Invoke all registered callbacks."""
        lock = obj._action_locks[self.name]
        with lock:
            callbacks = list(obj._action_callbacks[self.name])

        for cb in callbacks:
            try:
                cb()
            except Exception:
                logger.warning(
                    "Action callback %r for '%s' raised an exception",
                    cb, self.name, exc_info=True,
                )

    # -- Callback management -------------------------------------------------

    def add_callback(self, obj: Any, callback: Callable) -> None:
        with obj._action_locks[self.name]:
            if callback not in obj._action_callbacks[self.name]:
                obj._action_callbacks[self.name].append(callback)

    def remove_callback(self, obj: Any, callback: Callable) -> None:
        with obj._action_locks[self.name]:
            try:
                obj._action_callbacks[self.name].remove(callback)
            except ValueError:
                pass

    # -- Repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"description={self.description!r}"]
        if not self.visible:
            parts.append("visible=False")
        return f"Action({', '.join(parts)})"
