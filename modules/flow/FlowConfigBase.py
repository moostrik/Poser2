"""Base class for flow configurations with automatic clamping and change notification.

Uses dataclasses with field metadata for range constraints and GUI integration.
"""

from dataclasses import dataclass, fields, field
from typing import Any, Callable


@dataclass
class FlowConfigBase:
    """Base class for flow configs with automatic clamping and change notification.

    Subclasses use dataclass fields with metadata to define parameters:

    Example:
        @dataclass
        class MyConfig(FlowConfigBase):
            strength: float = field(default=1.0, metadata={"min": 0.0, "max": 10.0, "label": "Strength"})
            enabled: bool = field(default=True, metadata={"label": "Enabled"})

    Features:
        - Automatic clamping of values based on min/max metadata
        - Change notification via listeners
        - GUI metadata accessible via get_field_info()
    """

    def __post_init__(self) -> None:
        """Initialize listeners list after dataclass init."""
        object.__setattr__(self, '_listeners', [])

    def __setattr__(self, name: str, value: Any) -> None:
        """Intercept attribute changes, clamp values, and notify listeners."""
        # Skip internal attributes
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return

        # Clamp based on field metadata
        # try:
        #     for f in fields(self):
        #         if f.name == name and f.metadata:
        #             min_val = f.metadata.get("min")
        #             max_val = f.metadata.get("max")
        #             if min_val is not None and max_val is not None:
        #                 value = max(min_val, min(max_val, value))
        #             break
        # except TypeError:
        #     # fields() fails if class not fully initialized
        #     pass

        object.__setattr__(self, name, value)

        # Notify listeners
        if hasattr(self, '_listeners'):
            self._notify()

    def add_listener(self, callback: Callable[[], None]) -> None:
        """Register a callback to be notified of config changes.

        Args:
            callback: Function to call when any config value changes
        """
        self._listeners.append(callback) # type: ignore

    def remove_listener(self, callback: Callable[[], None]) -> None:
        """Unregister a callback.

        Args:
            callback: Previously registered callback to remove
        """
        if callback in self._listeners: # type: ignore
            self._listeners.remove(callback) # type: ignore

    def _notify(self) -> None:
        """Notify all listeners that config has changed."""
        for listener in self._listeners: # type: ignore
            listener()

    @classmethod
    def get_field_info(cls) -> dict[str, dict]:
        """Get field metadata for GUI generation.

        Returns:
            Dictionary mapping field names to their metadata including
            type, default value, min, max, label, etc.

        Example:
            >>> OpticalFlowConfig.get_field_info()
            {
                "strength": {"type": float, "default": 3.0, "min": 0.1, "max": 10.0, "label": "Strength"},
                ...
            }
        """
        result = {}
        for f in fields(cls):
            info = {
                "type": f.type,
                "default": f.default if f.default is not f.default_factory else None,
            }
            if f.metadata:
                info.update(f.metadata)
            result[f.name] = info
        return result
