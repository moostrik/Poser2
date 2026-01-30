"""Base class for flow configurations with change notification and GUI metadata.

Uses dataclasses with field metadata for range constraints and GUI integration.
"""

import threading
import warnings
from dataclasses import dataclass, fields, field, MISSING
from typing import Any, Callable, overload


def _generate_label(name: str) -> str:
    """Convert field name to human-readable label, preserving uppercase acronyms.

    Examples:
        "min_cutoff" -> "Min Cutoff"
        "TCP_PORT" -> "TCP PORT"
        "x" -> "X"
    """
    parts: list[str] = name.split('_')
    result: list[str] = []
    for part in parts:
        # Preserve all-caps acronyms (length > 1)
        if part.isupper() and len(part) > 1:
            result.append(part)
        else:
            result.append(part.capitalize())
    return ' '.join(result)


@dataclass
class ConfigBase:
    """Base class for flow configs with change notification and GUI metadata.

    Subclasses use dataclass fields with metadata to define parameters:

    Example:
        @dataclass
        class MyConfig(ConfigBase):
            enabled: bool = field(default=True)
            strength: int = field(default=1, metadata={"min": 0, "max": 10, "description": "MyConfig strength"})
            threshold: float = field(default=0.1, metadata={"min": 0.0, "max": 1.0, "description": "MyConfig threshold"})

    Features:
        - Thread-safe change notification via listeners
        - GUI metadata with auto-generated labels from field names
        - min/max values are hints for GUI, not enforced by this class
    """

    def __post_init__(self) -> None:
        """Initialize listeners, lock, and validate metadata."""
        object.__setattr__(self, '_listeners', set())
        object.__setattr__(self, '_lock', threading.Lock())

        # Validate metadata: check if default values are within min/max ranges
        for f in fields(self):
            if 'min' in f.metadata and 'max' in f.metadata:
                val = getattr(self, f.name)
                min_val, max_val = f.metadata['min'], f.metadata['max']
                if not (min_val <= val <= max_val):
                    warnings.warn(
                        f"{self.__class__.__name__}.{f.name}: default value {val} "
                        f"is outside valid range [{min_val}, {max_val}]",
                        UserWarning,
                        stacklevel=2
                    )

    def __setattr__(self, name: str, value: Any) -> None:
        """Intercept attribute changes and notify listeners.

        Note: Values are not clamped. min/max metadata are hints for GUI only.
        Only declared dataclass fields can be set.

        Raises:
            AttributeError: If attempting to set an undeclared attribute.
        """
        # Skip internal attributes
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return

        # Only allow setting declared dataclass fields
        field_names: set[str] = {f.name for f in fields(self)}
        if name not in field_names:
            raise AttributeError(
                f"Cannot set undeclared attribute '{name}' on {self.__class__.__name__}. "
                f"Only declared dataclass fields can be modified."
            )

        object.__setattr__(self, name, value)

        # Notify listeners
        if hasattr(self, '_listeners'):
            self._notify()

    def _notify(self) -> None:
        """Notify all listeners that config has changed."""
        with self._lock:  # type: ignore
            listeners_copy: list[Callable] = list(self._listeners)  # type: ignore

        for listener in listeners_copy:
            listener()

    @overload
    def watch(self, callback: Callable[[], None], attribute: None = None) -> Callable[[], None]:
        """Watch all config changes. Returns function to stop watching."""
        ...

    @overload
    def watch(self, callback: Callable[[Any], None], attribute: str) -> Callable[[], None]:
        """Watch specific attribute. Returns function to stop watching."""
        ...

    def watch(self, callback: Callable, attribute: str | None = None) -> Callable[[], None]:
        """Watch for config changes.

        Args:
            callback: Function to call when changes occur.
                     - If attribute is None: callback() with no arguments
                     - If attribute given: callback(value) with the new value
            attribute: Optional attribute name to watch (e.g., 'strength', 'threshold').
                      If None, watches all config changes.

        Returns:
            Function that when called, stops watching (removes the listener).

        Raises:
            AttributeError: If the specified attribute does not exist.

        Warning:
            Avoid creating infinite loops by modifying config values inside callbacks.
            When using with GUI frameworks like NiceGUI, prefer read-only operations
            in callbacks (e.g., updating UI elements) rather than writing back to config.

        Examples:
            >>> # Watch all changes
            >>> unwatch = config.watch(lambda: print("Config changed!"))
            >>> unwatch()  # Stop watching
            >>>
            >>> # Watch specific attribute
            >>> unwatch = config.watch(lambda val: print(f"Strength: {val}"), 'strength')
            >>> unwatch()  # Stop watching strength
            >>>
            >>> # Safe GUI pattern:
            >>> unwatch = config.watch(lambda val: slider.set_value(val), 'strength')
        """
        if attribute is None:
            # Watch all changes
            with self._lock:  # type: ignore
                self._listeners.add(callback)  # type: ignore

            # Return cleanup function
            def unwatch() -> None:
                with self._lock:  # type: ignore
                    self._listeners.discard(callback)  # type: ignore
            return unwatch
        else:
            # Validate attribute exists in declared fields
            field_names = {f.name for f in fields(self)}
            if attribute not in field_names:
                raise AttributeError(
                    f"Attribute '{attribute}' not found in {self.__class__.__name__}. "
                    f"Available attributes: {', '.join(sorted(field_names))}"
                )

            # Watch specific attribute
            def wrapper() -> None:
                callback(getattr(self, attribute))

            with self._lock:  # type: ignore
                self._listeners.add(wrapper)  # type: ignore

            # Return cleanup function that removes the wrapper
            def unwatch() -> None:
                with self._lock:  # type: ignore
                    self._listeners.discard(wrapper)  # type: ignore
            return unwatch

    @overload
    @classmethod
    def info(cls, attribute: None = None) -> dict[str, dict[str, Any]]:
        """Get metadata for all attributes."""
        ...

    @overload
    @classmethod
    def info(cls, attribute: str) -> dict[str, Any]:
        """Get metadata for a specific attribute."""
        ...

    @classmethod
    def info(cls, attribute: str | None = None) -> dict[str, Any] | dict[str, dict[str, Any]]:
        """Get config metadata for GUI generation.

        Args:
            attribute: Optional attribute name. If provided, returns metadata for that
                      attribute only. If None, returns metadata for all attributes.

        Returns:
            If attribute is None: Dictionary mapping all attribute names to their metadata
            If attribute given: Dictionary with metadata for that single attribute

            Metadata includes: type, default, min, max, label, description, etc.
            Labels are auto-generated from names (e.g., min_cutoff -> "Min Cutoff").

        Raises:
            AttributeError: If the specified attribute does not exist.

        Examples:
            >>> OpticalFlowConfig.info()
            {"strength": {"type": float, "default": 3.0, "min": 0.1, ...}, ...}
            >>> OpticalFlowConfig.info('strength')
            {"type": float, "default": 3.0, "min": 0.1, "max": 10.0, "label": "Strength", ...}
        """
        result: dict[str, dict[str, Any]] = {}
        for f in fields(cls):
            # Get default value properly
            if f.default is not MISSING:
                default_val = f.default
            elif f.default_factory is not MISSING:
                default_val = f.default_factory()
            else:
                default_val = None

            result[f.name] = {
                "type": f.type,
                "default": default_val,
                **f.metadata,
                "label": f.metadata.get("label", _generate_label(f.name)),
                "description": f.metadata.get("description", "")
            }

        # Return specific attribute or all
        if attribute is not None:
            if attribute not in result:
                raise AttributeError(f"Attribute '{attribute}' not found in {cls.__name__}")
            return result[attribute]
        return result
