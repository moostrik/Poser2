# Standard library imports
import hashlib
import inspect
import os
import sys
import time
import traceback
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec, spec_from_file_location
from types import CodeType, ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypedDict, get_type_hints

# Third-party imports
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

class MethodInfo(TypedDict):
    method: Callable
    params: List[Tuple[str, Type]]
    return_type: Type

MethodDict = Dict[str, MethodInfo]

class HotReloadStaticMethods:
    def __init__(self, target_class: Type[Any], methods_file_path: str, ) -> None:
        self._target_class: type = target_class
        self._methods_file_path: str = os.path.abspath(os.path.normcase(methods_file_path)).lower()
        self._method_types: MethodDict = self._collect_static_method_info(target_class)
        file_hash: str = hashlib.md5(self._methods_file_path.encode()).hexdigest()
        self._module_name: str = f"{self.__class__.__name__}_{file_hash}"
        self._observer: Optional[BaseObserver] = None
        self._cached_method_codes: Dict[str, Optional[CodeType]] = {}
        self._start_file_watcher()

    def _start_file_watcher(self) -> None:
        event_handler: HotReloadStaticMethods.FileChangeHandler = self.FileChangeHandler(self)
        self._observer = Observer()
        self._observer.schedule(event_handler, os.path.dirname(self._methods_file_path) or ".", recursive=False)
        self._observer.daemon = True
        self._observer.start()
        self.reload_methods(False)

    def reload_methods(self, verbose: bool = True) -> None:
        try:
            module: Optional[ModuleType] = HotReloadStaticMethods._import_module_from_file(self._module_name, self._methods_file_path)
            if module is None:
                return

            for name in self._method_types:
                if not hasattr(module, name):
                    print(f"[{HotReloadStaticMethods.__name__}] Method not found in module: {name}")
                    continue

                method: Callable = getattr(module, name)
                method_info: MethodInfo = self._method_types[name]
                method_sig: inspect.Signature = inspect.signature(method)

                if not HotReloadStaticMethods._check_parameter_names_match(name, method_sig, method_info['params']):
                    continue
                if not HotReloadStaticMethods._check_parameter_types_match(name, method_sig, method_info['params']):
                    continue
                if not HotReloadStaticMethods._check_return_type_match(name, method, method_info['return_type']):
                    continue

                changed, new_code = HotReloadStaticMethods._has_code_changed(name, method, self._cached_method_codes.get(name))
                if not changed or new_code is None:
                    continue
                self._cached_method_codes[name] = new_code

                setattr(self._target_class, name, staticmethod(method))
                if verbose:
                    print(f"[{HotReloadStaticMethods.__name__}] Patched method: {name} for {self._target_class.__name__}")

        except Exception as e:
            print(f"[MethodReloader] Error reloading {self._methods_file_path}: {e}")
            traceback.print_exc()

    @staticmethod
    def _collect_static_method_info(target_class: Type[Any]) -> MethodDict:
        static_methods: MethodDict = {}
        for name, method in inspect.getmembers(target_class):
            if isinstance(inspect.getattr_static(target_class, name), staticmethod) and not name.startswith('__'):
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

    @staticmethod
    def _import_module_from_file(module_name: str, methods_file_path: str) -> Optional[ModuleType]:
        spec: Optional[ModuleSpec] = spec_from_file_location(module_name, methods_file_path)
        if spec is None or spec.loader is None:
            print(f"[{HotReloadStaticMethods.__name__}] Could not load spec from {methods_file_path}")
            return None

        module: ModuleType = module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"[{HotReloadStaticMethods.__name__}] Error executing module {methods_file_path}: {e}")
            traceback.print_exc()
            return None
        return module

    @staticmethod
    def _check_parameter_names_match(name: str, method_sig: inspect.Signature, expected_params: List[Tuple[str, Type]]) -> bool:
        if len(method_sig.parameters) != len(expected_params):
            method_param_names: List[str] = list(method_sig.parameters.keys())
            expected_param_names: List[str] = [param[0] for param in expected_params]

            missing_params: List[str] = [p for p in expected_param_names if p not in method_param_names]
            extra_params: List[str] = [p for p in method_param_names if p not in expected_param_names]

            error_msg = f"[{HotReloadStaticMethods.__name__}] Error: {name} has {len(method_sig.parameters)} parameters, expected {len(expected_params)}."
            if missing_params:
                error_msg += f" Missing: {', '.join(missing_params)}."
            if extra_params:
                error_msg += f" Extra: {', '.join(extra_params)}."

            print(error_msg)
            return False
        return True

    @staticmethod
    def _check_parameter_types_match(name: str, method_sig: inspect.Signature, expected_params: List[Tuple[str, Type]]) -> bool:
        for i, (param_name, param) in enumerate(method_sig.parameters.items()):
            expected_name, expected_type = expected_params[i]
            if param.annotation != expected_type and param.annotation != inspect.Parameter.empty:
                print(f"[{HotReloadStaticMethods.__name__}] Error: {name} parameter {param_name} type {param.annotation} doesn't match expected {expected_type}")
                return False

        return True

    @staticmethod
    def _check_return_type_match(name: str, method: Callable, expected_return: Type) -> bool:
        method_hints: dict[str, Type] = get_type_hints(method)
        if 'return' in method_hints and method_hints['return'] != expected_return:
            print(f"[{HotReloadStaticMethods.__name__}] Error: {name} return type {method_hints['return']} doesn't match expected {expected_return}")
            return False
        return True

    @staticmethod
    def _has_code_changed(name: str, method: Callable, last_code: Optional[CodeType]) -> Tuple[bool, Optional[CodeType]]:
        new_code: Optional[CodeType] = getattr(method, "__code__", None)
        if not new_code:
            print(f"[{HotReloadStaticMethods.__name__}] Error: {name} has no code object")
            return False, None

        # If we have no previous code, definitely changed
        if last_code is None:
            return True, new_code

        # Check if code content has changed
        # Compare bytecode
        if new_code.co_code != last_code.co_code:
            print(f"[{HotReloadStaticMethods.__name__}] Bytecode changed for {name}")
            return True, new_code

        # Compare constants (could contain different values)
        if new_code.co_consts != last_code.co_consts:
            print(f"[{HotReloadStaticMethods.__name__}] Constants changed for {name}")
            return True, new_code

        # Compare names (variable names, etc.)
        if new_code.co_names != last_code.co_names:
            print(f"[{HotReloadStaticMethods.__name__}] Names changed for {name}")
            return True, new_code

        # Compare variable names
        if new_code.co_varnames != last_code.co_varnames:
            print(f"[{HotReloadStaticMethods.__name__}] Variable names changed for {name}")
            return True, new_code

        # Code hasn't changed, no need to patch
        return False, None

    class FileChangeHandler(FileSystemEventHandler):
        def __init__(self, reloader: 'HotReloadStaticMethods') -> None:
            self._reloader: HotReloadStaticMethods = reloader
            self._last_modified_time: float = 0
            self._debounce_seconds: float = 0.5  # Wait 500ms between reloads

        def on_modified(self, event: FileSystemEvent) -> None:
            current_time: float = time.time()
            if current_time - self._last_modified_time < self._debounce_seconds:
                return

            src_path: str = ""
            if isinstance(event.src_path, bytes):
                src_path = event.src_path.decode('utf-8')
            if isinstance(event.src_path, str):
                src_path = event.src_path

            event_path: str = os.path.abspath(os.path.normcase(src_path)).lower()
            watched_path: str = os.path.abspath(os.path.normcase(self._reloader._methods_file_path)).lower()
            if event_path == watched_path:
                self._last_modified_time = current_time
                self._reloader.reload_methods()

