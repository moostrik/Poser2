"""Action descriptor — a stateless trigger with callbacks."""

import logging
import threading

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
        description="",
        visible=True,
    ):
        self.description = description
        self.visible = visible
        self.name = ""

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self

    def __set__(self, obj, value):
        raise AttributeError(
            f"Action '{self.name}' is not assignable — call fire() instead"
        )

    # -- Fire ----------------------------------------------------------------

    def fire(self, obj):
        """Invoke all registered callbacks."""
        lock = obj._locks[self.name]
        with lock:
            callbacks = list(obj._callbacks[self.name])

        for cb in callbacks:
            try:
                cb()
            except Exception:
                logger.warning(
                    "Action callback %r for '%s' raised an exception",
                    cb, self.name, exc_info=True,
                )

    # -- Callback management -------------------------------------------------

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

    # -- Repr ----------------------------------------------------------------

    def __repr__(self):
        parts = [f"description={self.description!r}"]
        if not self.visible:
            parts.append("visible=False")
        return f"Action({', '.join(parts)})"
