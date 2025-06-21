from importlib.machinery import ModuleSpec
import importlib.util
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, TypedDict, Tuple
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from types import ModuleType, CodeType
import os
import inspect
import hashlib
from typing import get_type_hints

from watchdog.observers.api import BaseObserver

class MethodInfo(TypedDict):
    method: Callable
    params: List[Tuple[str, Type]]
    return_type: Type

MethodDict = Dict[str, MethodInfo]

class HotReloadStaticMethods:
    def __init__(self, target_class: Type[Any], methods_file_path: str, ) -> None:
        self.methods_file_path: str = os.path.abspath(os.path.normcase(methods_file_path)).lower()
        self.target_class: type = target_class
        self.method_types: MethodDict = self._get_static_methods_with_types(target_class)

        self.method_names: List[str] = list(self.method_types.keys())
        unique_hash: str = hashlib.md5(self.methods_file_path.encode()).hexdigest()
        self.module_name: str = f"{self.__class__.__name__}_{unique_hash}"
        self.observer: Optional[BaseObserver] = None
        self._last_method_codes: Dict[str, Optional[CodeType]] = {}
        self._start_watching()

    def _get_static_methods_with_types(self, cls: Type[Any]) -> MethodDict:
        static_methods: MethodDict = {}
        for name, method in inspect.getmembers(cls):
            if isinstance(inspect.getattr_static(cls, name), staticmethod) and not name.startswith('__'):
                sig: inspect.Signature = inspect.signature(method)
                params: list[tuple[str, inspect.Parameter]] = list(sig.parameters.items())
                param_types: list[tuple[str, Type]] = [(name, param.annotation) for name, param in params]
                return_type: Type = sig.return_annotation
                static_methods[name] = {
                    'method': method,
                    'params': param_types,
                    'return_type': return_type
                }
        return static_methods

    def _start_watching(self) -> None:
        event_handler: HotReloadStaticMethods.ReloadHandler = self.ReloadHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, os.path.dirname(self.methods_file_path) or ".", recursive=False)
        self.observer.daemon = True
        self.observer.start()
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
                    method: Callable = getattr(module, name)

                    # Validate method signature
                    try:
                        method_info: MethodInfo = self.method_types[name]
                        method_sig: inspect.Signature = inspect.signature(method)

                        # Check parameter count
                        expected_params: List[Tuple[str, Type]] = method_info['params']
                        if len(method_sig.parameters) != len(expected_params):
                            method_param_names: List[str] = list(method_sig.parameters.keys())
                            expected_param_names: List[str] = [param[0] for param in expected_params]

                            missing_params: List[str] = [p for p in expected_param_names if p not in method_param_names]
                            extra_params: List[str] = [p for p in method_param_names if p not in expected_param_names]

                            error_msg = f"[MethodReloader] Error: {name} has {len(method_sig.parameters)} parameters, expected {len(expected_params)}."
                            if missing_params:
                                error_msg += f" Missing: {', '.join(missing_params)}."
                            if extra_params:
                                error_msg += f" Extra: {', '.join(extra_params)}."

                            print(error_msg)
                            continue  # Skip patching this method

                        # Check return type
                        expected_return: Type = method_info['return_type']
                        method_hints: dict[str, Type] = get_type_hints(method)
                        if 'return' in method_hints and method_hints['return'] != expected_return:
                            print(f"[MethodReloader] Error: {name} return type {method_hints['return']} doesn't match expected {expected_return}")
                            continue  # Skip patching this method

                        # Check parameter types
                        for i, (param_name, param) in enumerate(method_sig.parameters.items()):
                            if i < len(expected_params):
                                expected_name, expected_type = expected_params[i]
                                if param.annotation != expected_type and param.annotation != inspect.Parameter.empty:
                                    print(f"[MethodReloader] Error: {name} parameter {param_name} type {param.annotation} doesn't match expected {expected_type}")
                                    continue  # Skip patching this method

                        # Only patch if method code has changed
                        new_code: Optional[CodeType] = getattr(method, "__code__", None)
                        last_code: Optional[CodeType] = self._last_method_codes.get(name)
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
        def __init__(self, reloader: 'HotReloadStaticMethods') -> None:
            self.reloader: HotReloadStaticMethods = reloader

        def on_modified(self, event: FileSystemEvent) -> None:
            src_path: str = ""
            if isinstance(event.src_path, bytes):
                src_path = event.src_path.decode('utf-8')
            if isinstance(event.src_path, str):
                src_path = event.src_path
            event_path: str = os.path.abspath(os.path.normcase(src_path)).lower()
            watched_path: str = os.path.abspath(os.path.normcase(self.reloader.methods_file_path)).lower()
            if event_path == watched_path:
                self.reloader.reload_methods()

