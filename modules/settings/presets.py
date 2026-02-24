"""Preset persistence — save / load / scan named settings files."""

import json
import logging
from pathlib import Path

from modules.settings.base_settings import BaseSettings

logger = logging.getLogger(__name__)

SETTINGS_DIR = Path("files/settings")
PRESET_SUFFIX = ".reactive.json"


def path(name: str) -> Path:
    """Return the full file path for a preset by name."""
    return SETTINGS_DIR / f"{name}{PRESET_SUFFIX}"


def scan() -> list[str]:
    """Return sorted list of preset names (without suffix) from the settings directory."""
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        p.name.removesuffix(PRESET_SUFFIX)
        for p in SETTINGS_DIR.glob(f"*{PRESET_SUFFIX}")
    )


def get_startup() -> str:
    """Return the preset name to load on startup (falls back to 'default')."""
    try:
        return (SETTINGS_DIR / "_startup_preset.txt").read_text().strip() or "default"
    except FileNotFoundError:
        return "default"


def set_startup(name: str) -> None:
    """Persist *name* as the preset to load on next startup."""
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    (SETTINGS_DIR / "_startup_preset.txt").write_text(name)


def save(root: BaseSettings, filepath) -> None:
    """Serialize all settings to a JSON file."""
    data = root.to_dict()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load(root: BaseSettings, filepath) -> None:
    """Restore settings from a JSON file.

    Skips unknown fields and init_only fields.  Silently handles
    corrupt or missing files.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    root.update_from_dict(data)


def load_startup(root: BaseSettings) -> None:
    """Load the startup preset.  Falls back to saving defaults if missing."""
    name = get_startup()
    p = path(name)
    if p.exists():
        load(root, p)
    else:
        save(root, path("default"))
