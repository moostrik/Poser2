import importlib
from pathlib import Path

# Every folder with a main.py is an app. Only the names are found here; an app is imported when it
# is loaded, so starting one never imports the others or their dependencies.
APP_NAMES: list[str] = sorted(p.name for p in Path(__file__).parent.iterdir()
                              if p.is_dir() and (p / "main.py").exists())


def load_app(name: str) -> type:
    """Import the app `name`'s main module and return its `*Main` class."""
    module = importlib.import_module(f"apps.{name}.main")
    class_name = next(k for k in vars(module) if k.endswith("Main"))
    return getattr(module, class_name)
