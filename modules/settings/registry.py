"""SettingsRegistry — pure data container for named BaseSettings modules."""

import logging

from modules.settings.base_settings import BaseSettings

logger = logging.getLogger(__name__)


class SettingsRegistry:
    """Stores multiple BaseSettings instances organised by group.

    This is a pure data container — all file I/O lives in ``presets``.
    """

    def __init__(self):
        self._modules = {}
        self._groups = {}

    def register(self, name, settings, group="default"):
        """Register a settings module under a group."""
        self._modules[name] = settings
        if group not in self._groups:
            self._groups[group] = []
        if name not in self._groups[group]:
            self._groups[group].append(name)

    def get(self, name):
        """Retrieve a registered settings module."""
        return self._modules[name]

    def groups(self):
        """Return a copy of {group: [config_names]} mapping."""
        return {g: list(names) for g, names in self._groups.items()}

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
        return f"SettingsRegistry(groups={dict(self._groups)})"
