import importlib
from pathlib import Path

APP_REGISTRY: dict[str, type] = {}
for _p in sorted(Path(__file__).parent.iterdir()):
    if _p.is_dir() and (_p / "main.py").exists():
        _mod = importlib.import_module(f"apps.{_p.name}")
        _cls_name = next(k for k in vars(_mod) if k.endswith("Main"))
        APP_REGISTRY[_p.name] = getattr(_mod, _cls_name)
