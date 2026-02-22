"""SettingsRegistry — stores multiple BaseSettings instances with JSON persistence."""

import json
from pathlib import Path

from modules.settings.BaseSettings import BaseSettings


class SettingsRegistry:
    """Stores multiple BaseSettings instances with group organization and JSON persistence."""

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

    def save(self, path):
        """Save all modules to a JSON file."""
        data = {name: settings.to_dict() for name, settings in self._modules.items()}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        """Load settings from a JSON file.

        Skips unknown modules and init_only fields. Silently handles
        corrupt or missing files.
        """
        path = Path(path)
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        for name, module_data in data.items():
            if name in self._modules:
                self._modules[name].update_from_dict(module_data)

    def __contains__(self, name):
        return name in self._modules

    def __getitem__(self, name):
        return self._modules[name]

    def __repr__(self):
        return f"SettingsRegistry(groups={dict(self._groups)})"
