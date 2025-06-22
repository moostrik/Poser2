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

class HotReloadStaticMethods:
    def __init__(self, target_class: Type[Any], watch_file: bool = True) -> None:
        if not inspect.isclass(target_class):
            raise ValueError(f"Expected a class, got {type(target_class).__name__}")
        self._target_class: Type[Any] = target_class

        class_module: Optional[ModuleType] = inspect.getmodule(target_class)
        if class_module is None or class_module.__file__ is None:
            raise ValueError(f"Could not determine module for class {target_class.__name__}")
        self._file_module_path: str = os.path.abspath(os.path.normcase(class_module.__file__)).lower()

        self._file_module_name: str = f"{self._target_class.__name__}_{hashlib.md5(self._file_module_path.encode()).hexdigest()}"
        file_module: Optional[ModuleType]  = self._load_module(self._file_module_name, self._file_module_path)

        if file_module is None:
            raise ValueError(f"Could not load module {self._file_module_name} from {self._file_module_path}")
        methods: Optional[Dict[str, Callable]] = self._get_static_methods_from_class(target_class)

        self._cached_method_code: Dict[str, CodeType] = {}
        if methods:
            self._cached_method_code: Dict[str, CodeType] = self._get_code_from_methods(methods)

        self._observer: Optional[BaseObserver] = None
        if watch_file:
            self._start_file_watcher()

    def _start_file_watcher(self) -> None:
        event_handler: HotReloadStaticMethods.FileChangeHandler = self.FileChangeHandler(self)
        self._observer = Observer()
        self._observer.schedule(event_handler, os.path.dirname(self._file_module_path) or ".", recursive=False)
        self._observer.daemon = True
        self._observer.start()

    def reload_methods(self) -> None:
        try:
            # Load the module from file
            module: Optional[ModuleType] = self._load_module(self._file_module_name, self._file_module_path)
            if module is None:
                return

            # Get static methods from the updated class in the module
            methods: Optional[Dict[str, Callable]] = self._get_static_methods_from_module(module, self._target_class.__name__)
            if methods is None:
                return

            new_method_code: Dict[str, CodeType] = self._get_code_from_methods(methods)

            # remove methods that no longer exist
            for name in list(self._cached_method_code.keys()):
                if name not in new_method_code:
                    print(f"[{HotReloadStaticMethods.__name__}] {self._target_class.__name__} remove method: {name}")
                    if hasattr(self._target_class, name):
                        method: Callable = getattr(self._target_class, name)
                        if isinstance(inspect.getattr_static(self._target_class, name), staticmethod):
                            print(f"[{HotReloadStaticMethods.__name__}] {self._target_class.__name__} removed method: {name}")
                            delattr(self._target_class, name)
                        else:
                            print(f"[{HotReloadStaticMethods.__name__}] {self._target_class.__name__} method {name} is not a static method, skipping removal")

            # update existing methods
            for name, code in new_method_code.items():
                if name in self._cached_method_code:
                    # print(f"[{HotReloadStaticMethods.__name__}] Checking method: {name} in {self._target_class.__name__}")
                    if self._has_code_changed(name, code, self._cached_method_code[name]):
                        if hasattr(self._target_class, name):
                            if isinstance(inspect.getattr_static(self._target_class, name), staticmethod):
                                print(f"[{HotReloadStaticMethods.__name__}] {self._target_class.__name__} patch method: {name}")
                                setattr(self._target_class, name, staticmethod(methods[name]))

            # add new methods that were added
            for name, code in new_method_code.items():
                if name not in self._cached_method_code:
                    print(f"[{HotReloadStaticMethods.__name__}] {self._target_class.__name__} add method: {name}")
                    setattr(self._target_class, name, staticmethod(methods[name]))

            # Update cached method codes
            self._cached_method_code = new_method_code

        except Exception as e:
            print(f"[{HotReloadStaticMethods.__name__}] {self._target_class.__name__} Error loading {self._file_module_path}: {e}")
            traceback.print_exc()

    @staticmethod
    def _load_module(module_name: str, file_path: str) -> Optional[ModuleType]:
        """Load a Python module from a file path."""
        spec: Optional[ModuleSpec] = spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            print(f"[{HotReloadStaticMethods.__name__}] Could not load spec from {file_path}")
            return None

        module: ModuleType = module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
            return module

        except Exception as e:
            print(f"[{HotReloadStaticMethods.__name__}] Error executing module {file_path}: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def _get_static_methods_from_class(obj: Type[Any]) -> Dict[str, Callable]:
        """Extract static methods from a class."""
        methods: Dict[str, Callable] = {}
        for name, method in inspect.getmembers(obj):
            if isinstance(inspect.getattr_static(obj, name), staticmethod) and not name.startswith('__'):
                methods[name] = method
        return methods

    @staticmethod
    def _get_static_methods_from_module(module: ModuleType, class_name: str) -> Optional[Dict[str, Callable]]:
        # Check if the class exists in the module
        if not hasattr(module, class_name):
            print(f"[{HotReloadStaticMethods.__name__}] Class {class_name} not found in module")
            return None
        # Get the class from the module
        class_obj = getattr(module, class_name)
        if not inspect.isclass(class_obj):
            print(f"[{HotReloadStaticMethods.__name__}] {class_name} is not a class in module {module.__name__}")
            return None

        return HotReloadStaticMethods._get_static_methods_from_class(class_obj)

    @staticmethod
    def _get_code_from_methods(methods: Dict[str, Callable]) -> Dict[str, CodeType]:
        """Extract code objects from static methods."""
        code_dict: Dict[str, CodeType] = {}
        for name, method in methods.items():
            code: Optional[CodeType] = getattr(method, "__code__", None)
            if code is not None:
                code_dict[name] = code
            else:
                print(f"[{HotReloadStaticMethods.__name__}] Method {name} has no code object")
        return code_dict

    @staticmethod
    def _has_code_changed(name: str, new_code: Optional[CodeType], last_code: Optional[CodeType]) -> bool:
        if not new_code:
            print(f"[{HotReloadStaticMethods.__name__}] Error: {name} has no code object")
            return False

        # If we have no previous code, definitely changed
        if last_code is None:
            return True

        # Check if code content has changed
        # Compare bytecode
        if new_code.co_code != last_code.co_code:
            # print(f"[{HotReloadStaticMethods.__name__}] Bytecode changed for {name}")
            return True

        # Compare constants (could contain different values)
        if new_code.co_consts != last_code.co_consts:
            # print(f"[{HotReloadStaticMethods.__name__}] Constants changed for {name}")
            return True

        # Compare names (variable names, etc.)
        if new_code.co_names != last_code.co_names:
            # print(f"[{HotReloadStaticMethods.__name__}] Names changed for {name}")
            return True

        # Compare variable names
        if new_code.co_varnames != last_code.co_varnames:
            # print(f"[{HotReloadStaticMethods.__name__}] Variable names changed for {name}")
            return True

        # Code hasn't changed, no need to patch
        return False

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
            watched_path: str = os.path.abspath(os.path.normcase(self._reloader._file_module_path)).lower()
            if event_path == watched_path:
                self._last_modified_time = current_time
                self._reloader.reload_methods()

