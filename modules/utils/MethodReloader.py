from importlib.machinery import ModuleSpec
import importlib.util
import sys
import traceback
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from types import ModuleType
import threading
import time

from watchdog.observers.api import BaseObserver

class MethodReloader:
    def __init__(self, methods_path: str, target_class: type , method_names: list[str]) -> None:
        """
        methods_path: path to the python file with methods to reload
        target_class: class to patch methods into (e.g., CompTest)
        method_names: list of method names (as strings) to reload from the file
        """
        self.methods_path: str = methods_path
        self.target_class: type = target_class
        self.method_names: list[str] = method_names
        self.module_name = "_hot_methods"
        self.observer: None | BaseObserver = None
        self._start_watching()

    def _start_watching(self) -> None:
        event_handler: MethodReloader.ReloadHandler = self.ReloadHandler(self)
        self.observer = Observer()
        import os
        self.observer.schedule(event_handler, os.path.dirname(self.methods_path) or ".", recursive=False)
        self.observer.daemon = True
        self.observer.start()
        # Initial load
        self.reload_methods()

    def reload_methods(self) -> None:
        try:
            spec: ModuleSpec | None = importlib.util.spec_from_file_location(self.module_name, self.methods_path)
            if spec is None or spec.loader is None:
                print(f"[MethodReloader] Could not load spec from {self.methods_path}")
                return
            module: ModuleType = importlib.util.module_from_spec(spec)
            sys.modules[self.module_name] = module
            spec.loader.exec_module(module)
            # Patch methods
            for name in self.method_names:
                if hasattr(module, name):
                    setattr(self.target_class, name, staticmethod(getattr(module, name)))
            print(f"[MethodReloader] Reloaded methods from {self.methods_path}")
        except Exception as e:
            print(f"[MethodReloader] Error reloading {self.methods_path}: {e}")
            traceback.print_exc()

    class ReloadHandler(FileSystemEventHandler):
        def __init__(self, reloader) -> None:
            self.reloader = reloader

        def on_modified(self, event) -> None:
            if event.src_path == self.reloader.methods_path:
                print(f"[MethodReloader] Detected change in {event.src_path}, reloading...")
                self.reloader.reload_methods()

# Example usage:
# reloader = MethodReloader("custom_methods.py", CompTest, ["make_fill", "make_pulse"])
# The above line will patch CompTest.make_fill and CompTest.make_pulse with the static methods from custom_methods.py