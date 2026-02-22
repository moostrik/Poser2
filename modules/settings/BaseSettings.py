"""BaseSettings — collection of Setting descriptors with attribute access, callbacks, and serialization."""

from __future__ import annotations

from typing import Any, Callable

import threading

from modules.settings.Setting import Setting
from modules.settings.Action import Action


class BaseSettings:
    """Collection of Setting descriptors with attribute access, callbacks, and serialization.

    Subclass and declare Setting descriptors as class attributes.
    Override init_only fields via constructor kwargs::

        class CameraSettings(BaseSettings):
            exposure = Setting(int, 1000, min=100, max=10000)
            resolution = Setting(int, 1080, init_only=True)

        settings = CameraSettings(resolution=720)
    """

    def __init__(self, **kwargs: Any) -> None:
        object.__setattr__(self, "_initialized", False)
        object.__setattr__(self, "_fields", {})
        object.__setattr__(self, "_values", {})
        object.__setattr__(self, "_callbacks", {})
        object.__setattr__(self, "_locks", {})
        object.__setattr__(self, "_actions", {})
        object.__setattr__(self, "_action_callbacks", {})
        object.__setattr__(self, "_action_locks", {})

        # Collect Setting and Action descriptors from the class hierarchy
        for cls in type(self).__mro__:
            for attr_name, attr_value in vars(cls).items():
                if isinstance(attr_value, Setting) and attr_name not in self._fields:
                    self._fields[attr_name] = attr_value
                    self._values[attr_name] = attr_value.default
                    self._callbacks[attr_name] = []
                    self._locks[attr_name] = threading.Lock()
                elif isinstance(attr_value, Action) and attr_name not in self._actions:
                    self._actions[attr_name] = attr_value
                    self._action_callbacks[attr_name] = []
                    self._action_locks[attr_name] = threading.Lock()

        # Apply kwargs (init_only fields are still writable here)
        for name, value in kwargs.items():
            if name not in self._fields:
                raise TypeError(
                    f"{type(self).__name__}() got unexpected keyword argument '{name}'"
                )
            self._fields[name].set(self, value)

        object.__setattr__(self, "_initialized", True)

    # -- Attribute access guard ----------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if name in self._actions:
            raise AttributeError(
                f"Action '{name}' is not assignable — call fire() instead"
            )
        if name not in self._fields:
            raise AttributeError(
                f"'{type(self).__name__}' has no setting '{name}'"
            )
        self._fields[name].set(self, value)

    def __getattr__(self, name: str) -> Any:
        # Fallback — normally the descriptor __get__ handles known fields.
        # This catches edge cases (e.g. dynamic lookup via getattr()).
        fields = object.__getattribute__(self, "_fields")
        if name in fields:
            return object.__getattribute__(self, "_values")[name]
        raise AttributeError(
            f"'{type(self).__name__}' has no setting '{name}'"
        )

    def __delattr__(self, name: str) -> None:
        if name.startswith("_"):
            object.__delattr__(self, name)
            return
        raise AttributeError(
            f"Cannot delete setting '{name}'"
        )

    # -- Field access --------------------------------------------------------

    @property
    def fields(self) -> dict[str, Setting]:
        """Read-only view of all Setting descriptors, keyed by name."""
        return dict(self._fields)

    @property
    def actions(self) -> dict[str, Action]:
        """Read-only view of all Action descriptors, keyed by name."""
        return dict(self._actions)

    # -- Callback registration -----------------------------------------------

    def remove_callback(self, name: str, callback: Callable) -> None:
        """Remove a previously registered callback for *name*."""
        if name not in self._fields:
            raise KeyError(f"Unknown setting '{name}'")
        self._fields[name].remove_callback(self, callback)

    def on_action(self, name: str, callback: Callable | None = None) -> Callable:
        """Register a callback for an action. Works as direct call or decorator."""
        if name not in self._actions:
            raise KeyError(f"Unknown action '{name}'")
        action = self._actions[name]
        if callback is not None:
            action.add_callback(self, callback)
            return callback
        def decorator(fn: Callable) -> Callable:
            action.add_callback(self, fn)
            return fn
        return decorator

    def remove_action_callback(self, name: str, callback: Callable) -> None:
        """Remove a previously registered action callback."""
        if name not in self._actions:
            raise KeyError(f"Unknown action '{name}'")
        self._actions[name].remove_callback(self, callback)

    def on_change(self, name: str, callback: Callable | None = None) -> Callable:
        """Register a callback for a field. Works as direct call or decorator.

        Direct::

            settings.on_change("exposure", my_callback)

        Decorator::

            @settings.on_change("exposure")
            def on_exposure(value): ...
        """
        if name not in self._fields:
            raise KeyError(f"Unknown setting '{name}'")

        field = self._fields[name]

        if callback is not None:
            field.add_callback(self, callback)
            return callback

        # Decorator mode
        def decorator(fn: Callable) -> Callable:
            field.add_callback(self, fn)
            return fn
        return decorator

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize mutable fields to a dict (excludes readonly and init_only)."""
        result = {}
        for name, field in self._fields.items():
            if not field.readonly and not field.init_only:
                result[name] = field.to_json_value(self)
        return result

    def update_from_dict(self, data: dict[str, Any]) -> None:
        """Restore fields from a dict. Skips init_only fields after init."""
        for name, raw in data.items():
            if name in self._fields:
                field = self._fields[name]
                if field.init_only and self._initialized:
                    continue
                field.from_json_value(self, raw)

    # -- Repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        parts = []
        for name, field in self._fields.items():
            if field.visible:
                value = self._values[name]
                parts.append(f"{name}={value!r}")
        class_name = type(self).__name__
        return f"{class_name}({', '.join(parts)})"
