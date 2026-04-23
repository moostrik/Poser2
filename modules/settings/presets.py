"""Preset persistence — save / load / scan named settings files."""

import json
import logging
from pathlib import Path

from .base_settings import BaseSettings

logger = logging.getLogger(__name__)

SETTINGS_DIR = Path("files/settings")
PRESET_SUFFIX = ".json"

_current_app: str | None = None


def set_app(app_name: str | None) -> None:
    """Set the active app name.  All subsequent preset calls use this app's directory."""
    global _current_app
    _current_app = app_name


def _app_dir(app_name: str | None = None) -> Path:
    """Return the settings directory, optionally scoped to an app."""
    name = app_name if app_name is not None else _current_app
    if name:
        return SETTINGS_DIR / name
    return SETTINGS_DIR


def path(name: str, app_name: str | None = None) -> Path:
    """Return the full file path for a preset by name."""
    return _app_dir(app_name) / f"{name}{PRESET_SUFFIX}"


def scan(app_name: str | None = None) -> list[str]:
    """Return sorted list of preset names (without suffix) from the settings directory."""
    d = _app_dir(app_name)
    d.mkdir(parents=True, exist_ok=True)
    return sorted(
        p.name.removesuffix(PRESET_SUFFIX)
        for p in d.glob(f"*{PRESET_SUFFIX}")
        if not p.name.startswith(".")
    )


def get_startup(app_name: str | None = None) -> str:
    """Return the preset name to load on startup (falls back to 'studio')."""
    try:
        data = json.loads((_app_dir(app_name) / ".ui_preset.json").read_text())
        return data.get("startup_preset") or "studio"
    except (FileNotFoundError, json.JSONDecodeError):
        return "studio"


def validate_name(name: str) -> None:
    """Raise ``ValueError`` if *name* is not a safe preset name.

    Rejects empty names, names containing path separators, and names
    starting with '.' — all of which could escape the settings directory.
    """
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"Invalid preset name: {name!r}")


def set_startup(name: str, app_name: str | None = None) -> None:
    """Persist *name* as the preset to load on next startup."""
    validate_name(name)
    d = _app_dir(app_name)
    d.mkdir(parents=True, exist_ok=True)
    (d / ".ui_preset.json").write_text(json.dumps({"startup_preset": name}, indent=2))


def save(root: BaseSettings, filepath) -> None:
    """Serialize all settings to a JSON file (atomic write)."""
    import os
    import tempfile

    data = root.to_dict()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file in the same directory, then atomically replace.
    fd, tmp = tempfile.mkstemp(dir=filepath.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, filepath)
    except BaseException:
        # Clean up the temp file on any failure
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def startup_path(app_name: str | None = None) -> Path:
    """Return the file path for the startup preset.

    If the file does not exist yet, a default preset is **not** created
    here — the caller decides what to do.
    """
    return path(get_startup(app_name), app_name)


def load(root: BaseSettings, filepath) -> bool:
    """Restore settings from a JSON file.

    Skips unknown fields and init_only fields.  Silently handles
    corrupt or missing files.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning("Preset file not found: %s", filepath)
        return False
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to load preset: %s", filepath)
        return False
    root.update_from_dict(data)
    logger.info("Loaded preset: %s", filepath)
    return True

