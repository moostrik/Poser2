"""Base class for flow configurations with change notification and GUI metadata.

Uses dataclasses with field metadata for range constraints and GUI integration.
"""

from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, fields, field, Field, MISSING
from typing import Any, Callable, overload, TypeVar

T = TypeVar('T')


# Valid metadata keys for config fields
METADATA_KEYS = {
    "description",  # Field description for tooltips/help
    "fixed",        # Field can be set at init, then becomes readonly
    "label",        # Custom display label (auto-generated if omitted)
    "min",          # Minimum value (GUI hint)
    "max",          # Maximum value (GUI hint)
}


# Helper function for creating config fields - for external use only
# Alternative names: cfield, cfg_field, meta_field, param
def config_field(
    default: T = MISSING,
    *,
    default_factory: Any = MISSING,
    description: str = "",
    label: str | None = None,
    min: float | int | None = None,
    max: float | int | None = None,
    fixed: bool = False,
    init: bool = True,
    repr: bool = True,
    hash: bool | None = None,
    compare: bool = True,
    kw_only: bool = False,
) -> T:
    """Create a config field with metadata.

    Note: Returns Field at runtime but typed as T for type checker compatibility.

    Args:
        default: Default value
        default_factory: Factory function for mutable defaults
        description: Field description for tooltips/help
        label: Custom display label (auto-generated if omitted)
        min: Minimum value hint for GUI (not enforced)
        max: Maximum value hint for GUI (not enforced)
        fixed: Field can be set during __init__, then becomes locked
        init: Include field in __init__ signature
        repr: Include field in __repr__ output
        hash: Include field in __hash__ calculation
        compare: Include field in comparison operations
        kw_only: Make field keyword-only in __init__

    Examples:
        >>> enabled: bool = config_field(True, description="Enable feature")
        >>> strength: float = config_field(1.0, min=0.0, max=10.0, description="Strength")
        >>> device_id: int = config_field(0, fixed=True, description="Camera device")
        >>> internal: int = config_field(0, repr=False, compare=False)
    """
    metadata = {}
    if description:
        metadata["description"] = description
    if label:
        metadata["label"] = label
    if min is not None:
        metadata["min"] = min
    if max is not None:
        metadata["max"] = max
    if fixed:
        metadata["fixed"] = True

    return field(  # type: ignore[return-value]
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        kw_only=kw_only,
        metadata=metadata
    )


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
            # Normal field - can change anytime
            enabled: bool = config_field(True, description="Enable feature")
            strength: float = config_field(1.0, min=0.0, max=10.0, description="Strength")

            # Fixed field - set at init, then locked
            device_id: int = config_field(0, fixed=True, description="Camera device ID")

    Metadata flags:
        - fixed: Field can be set during __init__, but becomes readonly after
        - min/max: Hints for GUI, not enforced
        - description: Field documentation
        - label: Custom display name (auto-generated from field name if omitted)

    Features:
        - Thread-safe change notification via listeners
        - GUI metadata with auto-generated labels from field names
        - Fixed fields are automatically hidden from GUI
    """

    def __post_init__(self) -> None:
        """Initialize listeners, lock, and validate metadata."""
        object.__setattr__(self, '_listeners', set())
        object.__setattr__(self, '_lock', threading.Lock())

        # Track which fields are fixed
        fixed_set: set[str] = set()
        for f in fields(self):
            # Validate metadata uses only known keys
            for key in f.metadata:
                if key not in METADATA_KEYS:
                    warnings.warn(
                        f"{self.__class__.__name__}.{f.name}: unknown metadata key '{key}' "
                        f"(valid keys: {', '.join(METADATA_KEYS)})",
                        UserWarning,
                        stacklevel=2
                    )

            # Add to fixed fields if fixed
            if f.metadata.get('fixed'):
                fixed_set.add(f.name)

            # Validate metadata: check if default values are within min/max ranges
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

        object.__setattr__(self, '_fixed_fields', fixed_set)
        object.__setattr__(self, '_initialized', True)

    def __setattr__(self, name: str, value: Any) -> None:
        """Intercept attribute changes for validation and notification.

        Only allows setting declared dataclass fields.
        Enforces fixed constraints after __post_init__.
        Notifies listeners after successful assignment.

        Raises:
            AttributeError: If field is undeclared, or fixed.
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
            )

        if not hasattr(self, '_initialized'):
            # During __init__, allow setting any attribute
            object.__setattr__(self, name, value)
            return

        # Check write constraints (after initialization)
        if name in self._fixed_fields: # type: ignore
            raise AttributeError(f"Cannot modify field '{name}'")

        # Thread-safe attribute setting (after initialization)
        with self._lock:  # type: ignore
            object.__setattr__(self, name, value)
            listeners_copy = list(self._listeners) # type: ignore

        # Notify listeners OUTSIDE lock to prevent deadlock
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
            Listeners are called OUTSIDE the lock to prevent deadlock.
            Avoid infinite loops - listeners can safely modify other config fields.
            For GUI updates, prefer read-only operations in callbacks.

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
                    f"Available attributes: {', '.join(field_names)}"
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
    def info(self, attribute: None = None) -> dict[str, dict[str, Any]]:
        """Get metadata for all attributes."""
        ...

    @overload
    def info(self, attribute: str) -> dict[str, Any]:
        """Get metadata for a specific attribute."""
        ...

    def info(self, attribute: str | None = None) -> dict[str, Any] | dict[str, dict[str, Any]]:
        """Get field metadata for GUI generation.

        Args:
            attribute: Optional field name. If provided, returns metadata for that
                      field only. If None, returns metadata for all fields.

        Returns:
            Dictionary with field metadata including both user-defined metadata
            and system-computed values (type, default, value).

            If attribute is None: {field_name: {metadata_dict}, ...}
            If attribute given: {metadata_dict}

        Raises:
            AttributeError: If the specified field does not exist.

        Examples:
            >>> config = OpticalFlowConfig()
            >>> config.info()
            {"strength": {"type": float, "default": 3.0, "value": 3.0, "min": 0.1, ...}, ...}
            >>> config.info('strength')
            {"type": float, "default": 3.0, "value": 3.0, "min": 0.1, "max": 10.0, ...}
        """
        result: dict[str, dict[str, Any]] = {}
        for f in fields(self):
            # Get default value properly
            if f.default is not MISSING:
                default_val = f.default
            elif f.default_factory is not MISSING:
                default_val = f.default_factory()
            else:
                default_val = None

            # User metadata first, then protected system keys override
            result[f.name] = {
                **f.metadata,
                "type": f.type,
                "default": default_val,
                "value": getattr(self, f.name),
            }
            # Add standard keys with defaults only if not present
            result[f.name].setdefault("label", _generate_label(f.name))
            result[f.name].setdefault("description", "")
            result[f.name].setdefault("min", None)
            result[f.name].setdefault("max", None)
            result[f.name].setdefault("fixed", False)

        # Return specific attribute or all
        if attribute is not None:
            if attribute not in result:
                raise AttributeError(f"Attribute '{attribute}' not found in {self.__class__.__name__}")
            return result[attribute]
        return result
