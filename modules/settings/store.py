"""The settings store: each root group's named files in its own folder of an app's settings folder."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from .base_settings import BaseSettings

logger = logging.getLogger(__name__)

SUFFIX = ".json"
STARTUP_FILE = ".startup.json"
SETUP = "setup"


class SettingsStore:
    """An app's settings folder: `<folder>/<group>/<name>.json` for each root group, and `.startup.json`, which
    names each root group's startup file; a root group it does not name starts on `default_startup`.

    The setup's file holds the root's own fields beside `setup`; another root group's file holds that group."""

    def __init__(self, folder: Path | str, default_startup: str = "studio") -> None:
        self.folder: Path = Path(folder)
        self.default_startup: str = default_startup

    # -- names and paths -----------------------------------------------------------

    @staticmethod
    def validate_name(name: str) -> None:
        """Raise ``ValueError`` for a name that is empty, holds a path separator or starts with '.'."""
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(f"Invalid settings file name: {name!r}")

    def path(self, group: str, name: str) -> Path:
        """The file of root group *group* named *name*."""
        return self.folder / group / f"{name}{SUFFIX}"

    def scan(self, group: str) -> list[str]:
        """The names of root group *group*'s files, sorted."""
        folder = self.folder / group
        if not folder.is_dir():
            return []
        return sorted(p.name.removesuffix(SUFFIX) for p in folder.glob(f"*{SUFFIX}") if not p.name.startswith("."))

    def delete(self, group: str, name: str) -> None:
        self.validate_name(name)
        self.path(group, name).unlink(missing_ok=True)

    # -- startup -------------------------------------------------------------------

    def _startup_names(self) -> dict[str, str]:
        try:
            data = json.loads((self.folder / STARTUP_FILE).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return {k: v for k, v in data.items() if isinstance(v, str)} if isinstance(data, dict) else {}

    def startup(self, group: str) -> str:
        """The name of root group *group*'s startup file."""
        return self._startup_names().get(group) or self.default_startup

    def set_startup(self, group: str, name: str) -> None:
        """Make *name* root group *group*'s startup file."""
        self.validate_name(name)
        names = self._startup_names()
        names[group] = name
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / STARTUP_FILE).write_text(json.dumps(names, indent=2), encoding="utf-8")

    # -- load and save -------------------------------------------------------------

    def load(self, root: BaseSettings, group: str, name: str) -> bool:
        """Load root group *group*'s file *name* into *root*; whether it was there and readable."""
        path = self.path(group, name)
        if not path.exists():
            logger.warning("Settings file not found: %s", path)
            return False
        try:
            data: Any = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Settings file unreadable: %s", path)
            return False
        if not isinstance(data, dict):
            logger.warning("Settings file holds no JSON object: %s", path)
            return False
        root.update_from_dict(data if group == SETUP else {group: data})
        logger.info("Loaded settings: %s", path)
        return True

    def save(self, root: BaseSettings, group: str, name: str) -> None:
        """Write root group *group* of *root* to its file *name*, atomically."""
        self.validate_name(name)
        tree = root.to_dict()
        if group == SETUP:
            data = {key: value for key, value in tree.items() if key not in root.children or key == SETUP}
        else:
            data = tree[group]
        path = self.path(group, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def load_startup(self, root: BaseSettings) -> None:
        """Load each root group's startup file, the setup's first. A root group without its file keeps its defaults,
        logged; the panel's Save then writes it."""
        for group in [SETUP] + [group for group in root.children if group != SETUP]:
            name = self.startup(group)
            if not self.load(root, group, name):
                logger.warning("No %s file '%s': %s starts on its defaults", group, name, group)
