"""SettingsRegistry — stores multiple BaseSettings instances with JSON persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modules.settings.base import BaseSettings


class SettingsRegistry:
    """Stores multiple BaseSettings instances. Handles save/load to JSON."""

    def __init__(self) -> None:
        self._modules: dict[str, BaseSettings] = {}

    def register(self, name: str, settings: BaseSettings) -> None:
        """Register a settings module."""
        self._modules[name] = settings

    def get(self, name: str) -> BaseSettings:
        """Retrieve a registered settings module."""
        return self._modules[name]

    def save(self, path: str | Path) -> None:
        """Save all modules to a JSON file."""
        data = {name: settings.to_dict() for name, settings in self._modules.items()}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | Path) -> None:
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

    def __contains__(self, name: str) -> bool:
        return name in self._modules

    def __getitem__(self, name: str) -> BaseSettings:
        return self._modules[name]

    def __repr__(self) -> str:
        names = list(self._modules.keys())
        return f"SettingsRegistry({names})"
