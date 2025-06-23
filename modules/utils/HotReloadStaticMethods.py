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
from typing import Any, Callable, Dict, List, Optional, Type

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

        methods: Optional[Dict[str, Callable]] = self._get_static_methods_from_class(target_class)
        self._cached_method_code: Dict[str, CodeType] = {}
        if methods:
            self._cached_method_code: Dict[str, CodeType] = self._get_code_from_methods(methods)

        self._on_reload_callbacks: list[Callable[[], None]] = []

        self._observer: Optional[BaseObserver] = None
        if watch_file:
            self.start_file_watcher()

    def start_file_watcher(self) -> None:
        """Start watching the file for changes."""
        if self._observer is not None:
            return

        event_handler: HotReloadStaticMethods.FileChangeHandler = self.FileChangeHandler(self)
        self._observer = Observer()
        self._observer.schedule(event_handler, os.path.dirname(self._file_module_path) or ".", recursive=False)
        self._observer.daemon = True
        self._observer.start()

    def stop_file_watcher(self) -> None:
        """Stop watching the file for changes."""
        if self._observer is None:
            return

        self._observer.stop()
        self._observer.join()  # Wait for the observer thread to finish
        self._observer = None

    def is_file_watcher_active(self) -> bool:
        """Check if the file watcher is active."""
        return self._observer is not None and self._observer.is_alive()

    def reload_methods(self) -> None:
        """Reload static methods from the target class's file and apply changes."""
        print(f"[{HotReloadStaticMethods.__name__}] Reloading static methods for {self._target_class.__name__} from {self._file_module_path}")
        try:
            # Load the module from the file path
            module: Optional[ModuleType] = self._load_module(self._file_module_name, self._file_module_path)
            if module is None:
                return

            # Get static methods from the module
            methods: Optional[Dict[str, Callable]] = self._get_static_methods_from_module(module, self._target_class.__name__)
            if methods is None:
                return

            # Get code objects from the static methods
            new_method_code: Dict[str, CodeType] = self._get_code_from_methods(methods)
            changed = False # Flag to track if any changes were made

            # Identify and remove deleted methods
            deleted_methods: List[str] = [name for name in self._cached_method_code if name not in new_method_code]
            if deleted_methods:
                HotReloadStaticMethods._remove_methods(self._target_class, deleted_methods)
                changed = True

            # Identify and update changed methods
            methods_to_update: Dict[str, Callable] = {}
            for name, code in new_method_code.items():
                if name in self._cached_method_code:
                    if HotReloadStaticMethods._is_different(code, self._cached_method_code[name]):
                        methods_to_update[name] = methods[name]
            if methods_to_update:
                HotReloadStaticMethods._update_methods(self._target_class, methods_to_update)
                changed = True

            # Identify and add new methods
            new_methods: Dict[str, Callable] = {name: methods[name] for name in new_method_code if name not in self._cached_method_code}
            if new_methods:
                HotReloadStaticMethods._add_methods(self._target_class, new_methods)
                changed = True

            # Update cached method codes
            self._cached_method_code = new_method_code

            if changed:
                self._notify_reload_callbacks()

        except Exception as e:
            print(f"[{HotReloadStaticMethods.__name__}] {self._target_class.__name__} Error loading {self._file_module_path}: {e}")
            traceback.print_exc()

    def add_reload_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called after code reload."""
        self._on_reload_callbacks.append(callback)

    def _notify_reload_callbacks(self) -> None:
        for callback in self._on_reload_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"[{self.__class__.__name__}] Error in reload callback: {e}")

    @staticmethod
    def _load_module(module_name: str, file_path: str) -> Optional[ModuleType]:
        """Load a module from a file path."""
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
        """Extract static methods from the class in a module."""
        if not hasattr(module, class_name):
            print(f"[{HotReloadStaticMethods.__name__}] Class {class_name} not found in module")
            return None
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
    def _is_different(new_code: Optional[CodeType], last_code: Optional[CodeType]) -> bool:
        """check if two code objects are different."""
        if new_code is None:
            return False
        if last_code is None:
            return True

        if new_code.co_code != last_code.co_code:
            return True
        if new_code.co_consts != last_code.co_consts:
            return True
        if new_code.co_names != last_code.co_names:
            return True
        if new_code.co_varnames != last_code.co_varnames:
            return True
        if new_code.co_nlocals != last_code.co_nlocals:
            return True
        if new_code.co_stacksize != last_code.co_stacksize:
            return True
        if new_code.co_flags != last_code.co_flags:
            return True
        if new_code.co_freevars != last_code.co_freevars:
            return True
        if new_code.co_cellvars != last_code.co_cellvars:
            return True

        return False

    @staticmethod
    def _remove_methods(target_class: Type[Any], deleted_methods: List[str]) -> None:
        """Remove methods that no longer exist in the updated code."""
        for name in deleted_methods:
            if not HotReloadStaticMethods._is_valid_static_method(target_class, name):
                print(f"[{HotReloadStaticMethods.__name__}] {target_class.__name__} method {name} is not a valid static method, skipping removal")
                continue

            print(f"[{HotReloadStaticMethods.__name__}] {target_class.__name__} remove method: {name}")
            delattr(target_class, name)

    @staticmethod
    def _update_methods(target_class: Type[Any], methods_to_update: Dict[str, Callable]) -> None:
        """Update methods that have changed."""
        for name, method in methods_to_update.items():
            if not HotReloadStaticMethods._is_valid_static_method(target_class, name):
                print(f"[{HotReloadStaticMethods.__name__}] {target_class.__name__} method {name} is not a valid static method, skipping update")
                continue

            print(f"[{HotReloadStaticMethods.__name__}] {target_class.__name__} patch method: {name}")
            setattr(target_class, name, staticmethod(method))

    @staticmethod
    def _add_methods(target_class: Type[Any], methods_to_add: Dict[str, Callable]) -> None:
        """Add methods that are new in the updated code."""
        for name, method in methods_to_add.items():
            if HotReloadStaticMethods._has_attribute_conflict(target_class, name):
                print(f"[{HotReloadStaticMethods.__name__}] {target_class.__name__} method {name} has conflict with class attribute")
                continue

            print(f"[{HotReloadStaticMethods.__name__}] {target_class.__name__} add method: {name}")
            setattr(target_class, name, staticmethod(method))

    @staticmethod
    def _is_valid_static_method(target_class: Type[Any], method_name: str) -> bool:
        """Check if the method is a valid static method."""
        if not hasattr(target_class, method_name):
            return False
        return isinstance(inspect.getattr_static(target_class, method_name), staticmethod) and not method_name.startswith('__')

    @staticmethod
    def _has_attribute_conflict(target_class: Type[Any], method_name: str) -> bool:
        """Note: this method does not work for instance attributes."""
        if hasattr(target_class, method_name):
            if not isinstance(inspect.getattr_static(target_class, method_name), staticmethod):
                return True
        return False

    class FileChangeHandler(FileSystemEventHandler):
        def __init__(self, reloader: 'HotReloadStaticMethods') -> None:
            self._reloader: HotReloadStaticMethods = reloader
            self._last_modified_time: float = 0
            self._debounce_seconds: float = 0.5  # Wait 500ms between reloads

        def on_modified(self, event: FileSystemEvent) -> None:
            """Handle file modification events."""
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

