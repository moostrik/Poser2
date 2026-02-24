"""Preset persistence — save / load / scan named settings files."""

import json
import logging
from pathlib import Path

from modules.settings.settings import Settings

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


def save(root: Settings, filepath) -> None:
    """Serialize all settings to a JSON file."""
    data = root.to_dict()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def startup_path() -> Path:
    """Return the file path for the startup preset.

    If the file does not exist yet, a default preset is **not** created
    here — the caller decides what to do.
    """
    return path(get_startup())


def load(root: Settings, filepath) -> bool:
    """Restore settings from a JSON file.

    Skips unknown fields and init_only fields.  Silently handles
    corrupt or missing files.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"Preset file not found: {filepath}")
        return False
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning(f"Failed to load preset: {filepath}")
        return False
    root.update_from_dict(data)
    logger.info(f"Loaded preset: {filepath}")
    return True

