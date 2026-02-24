"""SettingsRegistry — pure data container for named BaseSettings modules."""

import logging

from modules.settings.base_settings import BaseSettings

logger = logging.getLogger(__name__)


class SettingsRegistry:
    """Stores multiple BaseSettings instances by name (insertion-ordered).

    This is a pure data container — all file I/O lives in ``presets``.
    The panel derives tabs from each root's children.
    """

    def __init__(self):
        self._modules: dict[str, BaseSettings] = {}

    def register(self, name: str, settings: BaseSettings) -> None:
        """Register a settings module."""
        self._modules[name] = settings

    def get(self, name: str) -> BaseSettings:
        """Retrieve a registered settings module."""
        return self._modules[name]

    def modules(self) -> dict[str, BaseSettings]:
        """Return insertion-ordered ``{name: settings}`` (shallow copy)."""
        return dict(self._modules)

    def to_dict(self) -> dict:
        """Serialize every module to a plain dict."""
        return {name: settings.to_dict() for name, settings in self._modules.items()}

    def from_dict(self, data: dict) -> None:
        """Restore modules from a plain dict (as produced by ``to_dict``).

        Unknown module names are silently skipped.
        """
        for name, module_data in data.items():
            if name in self._modules:
                try:
                    self._modules[name].update_from_dict(module_data)
                except Exception:
                    logger.warning(
                        "Failed to load settings for '%s', using defaults", name,
                        exc_info=True,
                    )

    def __contains__(self, name):
        return name in self._modules

    def __getitem__(self, name):
        return self._modules[name]

    def __repr__(self):
        return f"SettingsRegistry(modules={list(self._modules)})"
