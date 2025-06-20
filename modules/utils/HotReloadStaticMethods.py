from importlib.machinery import ModuleSpec
import importlib.util
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Type
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from types import ModuleType
import os
import inspect
import hashlib
from typing import get_type_hints

from watchdog.observers.api import BaseObserver

class HotReloadStaticMethods:
    def __init__(self, methods_file_path: str, target_class: type, method_types: Dict[str, Type[Callable]]) -> None:
        self.methods_file_path: str = os.path.abspath(os.path.normcase(methods_file_path)).lower()
        self.target_class: type = target_class
        self.method_types: Dict[str, Type[Callable]] = method_types
        self.method_names: List[str] = list(method_types.keys())
        unique_hash: str = hashlib.md5(self.methods_file_path.encode()).hexdigest()
        self.module_name: str = f"{self.__class__.__name__}_{unique_hash}"
        self.observer: Optional[BaseObserver] = None
        self._last_method_codes: Dict[str, Any] = {}  # Store last code objects
        self._start_watching()

    def _start_watching(self) -> None:
        event_handler: HotReloadStaticMethods.ReloadHandler = self.ReloadHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, os.path.dirname(self.methods_file_path) or ".", recursive=False)
        self.observer.daemon = True
        self.observer.start()
        # Initial load
        self.reload_methods(False)

    def reload_methods(self, verbose: bool = True) -> None:
        try:
            spec: Optional[ModuleSpec] = importlib.util.spec_from_file_location(self.module_name, self.methods_file_path)
            if spec is None or spec.loader is None:
                print(f"[MethodReloader] Could not load spec from {self.methods_file_path}")
                return

            module: ModuleType = importlib.util.module_from_spec(spec)
            sys.modules[self.module_name] = module
            spec.loader.exec_module(module)

            # Patch methods
            for name in self.method_names:
                if hasattr(module, name):
                    method = getattr(module, name)

                    # Validate method signature
                    try:
                        expected_type = self.method_types[name]
                        method_sig: inspect.Signature = inspect.signature(method)
                        method_hints: Dict[str, Any] = get_type_hints(method)

                        # Check parameter count
                        param_count: int = len(method_sig.parameters)
                        expected_param_count: int = len(expected_type.__args__) - 1  # -1 for return type

                        if param_count != expected_param_count:
                            print(f"[MethodReloader] Error: {name} has {param_count} parameters, expected {expected_param_count}")
                            continue  # Skip patching this method

                        # Check return type if available
                        if 'return' in method_hints and method_hints['return'] != expected_type.__args__[-1]:
                            print(f"[MethodReloader] Error: {name} return type {method_hints['return']} doesn't match expected {expected_type.__args__[-1]}")
                            continue  # Skip patching this method

                        # Only patch if method code has changed
                        new_code = getattr(method, "__code__", None)
                        last_code = self._last_method_codes.get(name)
                        if new_code and new_code != last_code:
                            setattr(self.target_class, name, staticmethod(method))
                            self._last_method_codes[name] = new_code
                            if verbose:
                                print(f"[MethodReloader] Patched method: {name} for {self.target_class.__name__}")
                        # else:
                        #     print(f"[MethodReloader] Skipped patching {name} (no change)")

                    except (AttributeError, TypeError, ValueError) as type_error:
                        print(f"[MethodReloader] Error validating {name} signature: {type_error}")
                        # Do not patch methods with validation errors
                else:
                    print(f"[MethodReloader] Method not found in module: {name}")

            # print(f"[MethodReloader] Reloaded methods from {self.methods_file_path}")
        except Exception as e:
            print(f"[MethodReloader] Error reloading {self.methods_file_path}: {e}")
            traceback.print_exc()

    class ReloadHandler(FileSystemEventHandler):
        def __init__(self, reloader) -> None:
            self.reloader = reloader

        def on_modified(self, event) -> None:
            event_path = os.path.abspath(os.path.normcase(event.src_path)).lower()
            watched_path = os.path.abspath(os.path.normcase(self.reloader.methods_file_path)).lower()
            if event_path == watched_path:
                # print(f"[MethodReloader] Detected change in {event.src_path}, reloading...")
                self.reloader.reload_methods()

# Example usage:
# method_types = {
#     "make_fill": Callable[[int, float, float], np.ndarray],
#     "make_pulse": Callable[[int, float, float, float, float], np.ndarray]
# }
# reloader = MethodReloader("custom_methods.py", CompTest, method_types)