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
import time

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

    def reload_methods(self, verbose: bool = True) -> None:
        try:
            module: Optional[ModuleType] = self._load_module()
            if module is None:
                return

            for name in self.method_names:
                if not hasattr(module, name):
                    print(f"[MethodReloader] Method not found in module: {name}")
                    continue
                method: Callable = getattr(module, name)
                method_info: MethodInfo = self.method_types[name]
                method_sig: inspect.Signature = inspect.signature(method)
                if not self._validate_parameter_names(name, method_sig, method_info['params']):
                    continue
                if not self._validate_parameter_types(name, method_sig, method_info['params']):
                    continue
                if not self._validate_return_type(name, method, method_info['return_type']):
                    continue
                if not self._validate_code_changed(name, method):
                    continue
                setattr(self.target_class, name, staticmethod(method))
                if verbose:
                    print(f"[MethodReloader] Patched method: {name} for {self.target_class.__name__}")

        except Exception as e:
            print(f"[MethodReloader] Error reloading {self.methods_file_path}: {e}")
            traceback.print_exc()

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

    def _load_module(self) -> Optional[ModuleType]:
        spec: Optional[ModuleSpec] = importlib.util.spec_from_file_location(self.module_name, self.methods_file_path)
        if spec is None or spec.loader is None:
            print(f"[MethodReloader] Could not load spec from {self.methods_file_path}")
            return None

        module: ModuleType = importlib.util.module_from_spec(spec)
        sys.modules[self.module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"[MethodReloader] Error executing module {self.methods_file_path}: {e}")
            traceback.print_exc()
            return None
        return module

    def _validate_parameter_names(self, name: str, method_sig: inspect.Signature, expected_params: List[Tuple[str, Type]]) -> bool:
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
            return False
        return True

    def _validate_parameter_types(self, name: str, method_sig: inspect.Signature, expected_params: List[Tuple[str, Type]]) -> bool:
        for i, (param_name, param) in enumerate(method_sig.parameters.items()):
            expected_name, expected_type = expected_params[i]
            if param.annotation != expected_type and param.annotation != inspect.Parameter.empty:
                print(f"[MethodReloader] Error: {name} parameter {param_name} type {param.annotation} doesn't match expected {expected_type}")
                return False

        return True

    def _validate_return_type(self, name: str, method: Callable, expected_return: Type) -> bool:
        method_hints: dict[str, Type] = get_type_hints(method)
        if 'return' in method_hints and method_hints['return'] != expected_return:
            print(f"[MethodReloader] Error: {name} return type {method_hints['return']} doesn't match expected {expected_return}")
            return False
        return True

    def _validate_code_changed(self, name: str, method: Callable) -> bool:
        """Validate if the method's code has changed since last reload.
        Returns True if the code has changed or wasn't previously loaded."""
        new_code: Optional[CodeType] = getattr(method, "__code__", None)
        if not new_code:
            print(f"[MethodReloader] Error: {name} has no code object")
            return False

        last_code: Optional[CodeType] = self._last_method_codes.get(name)

        # If we have no previous code or the code has changed
        if last_code is None or self._code_content_changed(new_code, last_code):
            self._last_method_codes[name] = new_code
            return True

        # Code hasn't changed, no need to patch
        return False

    def _code_content_changed(self, new_code: CodeType, old_code: CodeType) -> bool:
        """Compare relevant attributes of code objects to determine if content changed."""
        # Compare bytecode
        if new_code.co_code != old_code.co_code:
            print(f"[MethodReloader] Bytecode changed for {new_code.co_name}")
            return True

        # # Compare constants (could contain different values)
        # if new_code.co_consts != old_code.co_consts:
        #     print(f"[MethodReloader] Constants changed for {new_code.co_name}")
        #     return True

        # Compare names (variable names, etc.)
        if new_code.co_names != old_code.co_names:
            print(f"[MethodReloader] Names changed for {new_code.co_name}")
            return True

        # Compare variable names
        if new_code.co_varnames != old_code.co_varnames:
            print(f"[MethodReloader] Variable names changed for {new_code.co_name}")
            return True

        return False

    class ReloadHandler(FileSystemEventHandler):
        def __init__(self, reloader: 'HotReloadStaticMethods') -> None:
            self.reloader: HotReloadStaticMethods = reloader
            self.last_modified_time: float = 0
            self.debounce_seconds: float = 0.5  # Wait 500ms between reloads

        def on_modified(self, event: FileSystemEvent) -> None:
            current_time: float = time.time()
            if current_time - self.last_modified_time < self.debounce_seconds:
                return

            src_path: str = ""
            if isinstance(event.src_path, bytes):
                src_path = event.src_path.decode('utf-8')
            if isinstance(event.src_path, str):
                src_path = event.src_path

            event_path: str = os.path.abspath(os.path.normcase(src_path)).lower()
            watched_path: str = os.path.abspath(os.path.normcase(self.reloader.methods_file_path)).lower()
            if event_path == watched_path:
                self.last_modified_time = current_time
                self.reloader.reload_methods()

