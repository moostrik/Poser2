"""Preset persistence — save / load / scan named settings files."""

import json
import logging
from pathlib import Path

from modules.settings.settings import Settings

logger = logging.getLogger(__name__)

SETTINGS_DIR = Path("files/settings")
PRESET_SUFFIX = ".json"


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
        data = json.loads((SETTINGS_DIR / ".ui_preset.json").read_text())
        return data.get("startup_preset") or "default"
    except (FileNotFoundError, json.JSONDecodeError):
        return "default"


def set_startup(name: str) -> None:
    """Persist *name* as the preset to load on next startup."""
    # Reject names that could escape the settings directory
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"Invalid preset name: {name!r}")
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    (SETTINGS_DIR / ".ui_preset.json").write_text(json.dumps({"startup_preset": name}, indent=2))


def save(root: Settings, filepath) -> None:
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

