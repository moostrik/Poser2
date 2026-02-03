# Standard library imports
import ast
import hashlib
import inspect
import os
import sys
import time
import traceback
from dataclasses import dataclass
from enum import Enum, auto
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec, spec_from_file_location
from types import CodeType, ModuleType
from typing import Any, Callable, ClassVar, Dict, List, Optional, Type

# Third-party imports
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

class MethodType(Enum):
    STATIC = auto()
    CLASS = auto()
    INSTANCE = auto()
    CONSTANT = auto()

@dataclass
class MethodInfo:
    type: MethodType
    func: Callable

MethodMap = Dict[str, MethodInfo] # Mapping of method names to MethodInfo

# wwhere

class HotReloadMethods:
    # Class-level registry: one watcher per file path
    _active_watchers: ClassVar[Dict[str, 'HotReloadMethods']] = {}

    def __init__(self, target_class: Type[Any], auto_reload: bool = True, reload_everything = True) -> None:
        if not inspect.isclass(target_class):
            raise ValueError(f"Expected a class, got {type(target_class).__name__}")
        self._target_class: Type[Any] = target_class

        class_module: Optional[ModuleType] = inspect.getmodule(target_class)
        if class_module is None or class_module.__file__ is None:
            raise ValueError(f"Could not determine module for class {target_class.__name__}")

        self._file_module_path: str = os.path.abspath(os.path.normcase(class_module.__file__)).lower()
        self._file_module_name: str = class_module.__name__
        self._module: ModuleType = class_module

        self._on_reload_callbacks: list[Callable[[], None]] = []

        self._observer: Optional[BaseObserver] = None

        self.auto_reload: bool = auto_reload
        self.reload_everything: bool = reload_everything
        self._has_file_changed: bool = False

        # Only create watcher if none exists for this file
        if self._file_module_path not in HotReloadMethods._active_watchers:
            self.start_file_watcher()
            HotReloadMethods._active_watchers[self._file_module_path] = self

    @property
    def file_changed(self) -> bool:
        return_value: bool = self._has_file_changed
        self._has_file_changed = False  # Reset after checking
        return return_value

    def start_file_watcher(self) -> None:
        """Start watching the file for changes."""
        if self._observer is not None:
            return

        event_handler: HotReloadMethods.FileChangeHandler = self.FileChangeHandler(self)
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

    def on_file_modified(self) -> None:
        """Handle file modification event."""
        # print(f"[{HotReloadMethods.__name__}] File modified: {self._file_module_path}")
        if self.auto_reload:
            self.reload_methods()

        self._has_file_changed = True  # Set the flag to indicate a file change
        self._notify_file_changed_callbacks()


    def reload_methods(self) -> None:
        """Reload methods and constants from the target class's file and apply changes."""
        # print(f"[{HotReloadMethods.__name__}] Reloading methods for {self._target_class.__name__} from {self._file_module_path}")
        try:
            # Get methods from the module
            module: Optional[ModuleType] = HotReloadMethods._load_module(self._file_module_name, self._file_module_path)
            if module is None:
                return
            module_methods: Optional[MethodMap] = HotReloadMethods._get_methods_from_module(module, self._target_class.__name__)
            if module_methods is None:
                return
            # Get methods from the target class
            class_methods: MethodMap = HotReloadMethods._get_methods_from_class(self._target_class)

            # Also get module-level constants
            module_constants: MethodMap = HotReloadMethods._get_module_constants(module)
            current_constants: MethodMap = HotReloadMethods._get_module_constants(self._module)

            changed = False

            # Identify and remove deleted methods, based on nyme and type
            # deleted_methods: List[str] = [name for name in class_methods if name not in module_methods or class_methods[name].type != module_methods[name].type]
            deleted_methods: Dict[str, MethodInfo] = {name: info for name, info in class_methods.items() if name not in module_methods or class_methods[name].type != module_methods[name].type}
            if deleted_methods:
                HotReloadMethods._remove_methods(self._target_class, deleted_methods)
                changed = True

            # Identify and update changed methods
            methods_to_update: MethodMap = {}
            for name, info in module_methods.items():
                if name in class_methods and class_methods[name].type == info.type:
                    if info.type == MethodType.CONSTANT:
                        # For constants, compare the values directly
                        if info.func != class_methods[name].func or self.reload_everything:
                            methods_to_update[name] = info
                    else:
                        # For methods, compare code objects
                        module_code: Optional[CodeType] = getattr(info.func, "__code__", None)
                        class_code: Optional[CodeType] = getattr(class_methods[name].func, "__code__", None)
                        if HotReloadMethods._is_different(module_code, class_code) or self.reload_everything:
                            methods_to_update[name] = info
            if methods_to_update:
                HotReloadMethods._update_methods(self._target_class, methods_to_update)
                changed = True

            # Identify and add new methods
            new_methods: Dict[str, MethodInfo] = {name: info for name, info in module_methods.items() if name not in class_methods or class_methods[name].type != info.type}
            if new_methods:
                HotReloadMethods._add_methods(self._target_class, new_methods)
                changed = True

            # Update module-level constants
            constants_to_update: MethodMap = {}
            for name, info in module_constants.items():
                if name in current_constants:
                    if info.func != current_constants[name].func or self.reload_everything:
                        constants_to_update[name] = info
                else:
                    # New constant
                    constants_to_update[name] = info

            if constants_to_update:
                HotReloadMethods._update_module_constants(self._module, constants_to_update)
                changed = True

            if changed:
                pass
                # self._notify_file_changed_callbacks()
            else:
                print(f"[{HotReloadMethods.__name__}] {self._target_class.__name__} No changes")

        except Exception as e:
            print(f"[{HotReloadMethods.__name__}] {self._target_class.__name__} Error loading {self._file_module_path}: {e}")
            traceback.print_exc()

    def add_file_changed_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called after code reload."""
        self._on_reload_callbacks.append(callback)

    def _notify_file_changed_callbacks(self) -> None:
        for callback in self._on_reload_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"[{self.__class__.__name__}] Error in reload callback: {e}")

    @staticmethod
    def _load_module(module_name: str, file_path: str) -> Optional[ModuleType]:
        """Load a module from a file path, executing only class definitions (not imports)."""
        try:
            # Read the source code
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            # Get the original module to use its namespace (for imports)
            original_module = sys.modules.get(module_name)
            if original_module is None:
                print(f"[{HotReloadMethods.__name__}] Original module {module_name} not found in sys.modules")
                return None

            # Parse the source to extract class definitions and module-level assignments
            tree = ast.parse(source_code, filename=file_path)

            # Filter to keep ClassDef nodes and module-level assignments (skip imports and function defs)
            filtered_nodes = []
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    filtered_nodes.append(node)
                elif isinstance(node, ast.Assign):
                    # Keep module-level assignments (constants)
                    filtered_nodes.append(node)
                elif isinstance(node, ast.AnnAssign):
                    # Keep annotated assignments
                    filtered_nodes.append(node)

            if not filtered_nodes:
                print(f"[{HotReloadMethods.__name__}] No class definitions or constants found in {file_path}")
                return None

            # Create a new module with only class definitions and constants
            tree.body = filtered_nodes # type: ignore

            # Compile and execute the filtered AST
            code = compile(tree, file_path, 'exec')

            # Execute in the original module's namespace directly (so imports work)
            exec(code, original_module.__dict__)

            return original_module

        except Exception as e:
            print(f"[{HotReloadMethods.__name__}] Error loading module {file_path}: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def _get_methods_from_class(obj: Type[Any]) -> MethodMap:
        """Extract all methods (static, class, instance) and constants from a class."""
        methods: MethodMap = {}
        for name, member in inspect.getmembers(obj):
            if name.startswith('__'):
                continue
            if name not in obj.__dict__:
                continue  # Skip inherited members
            attr = inspect.getattr_static(obj, name)
            if isinstance(attr, staticmethod):
                methods[name] = MethodInfo(MethodType.STATIC, member)
            elif isinstance(attr, classmethod):
                methods[name] = MethodInfo(MethodType.CLASS, member)
            elif inspect.isfunction(attr):
                methods[name] = MethodInfo(MethodType.INSTANCE, member)
            elif not callable(attr) and not inspect.ismethod(attr):
                # It's a constant (class variable)
                methods[name] = MethodInfo(MethodType.CONSTANT, attr)
        return methods

    @staticmethod
    def _get_methods_from_module(module: ModuleType, class_name: str) -> Optional[MethodMap]:
        """Extract methods from the class in a module."""
        if not hasattr(module, class_name):
            print(f"[{HotReloadMethods.__name__}] Class {class_name} not found in module")
            return None
        class_obj = getattr(module, class_name)
        if not inspect.isclass(class_obj):
            print(f"[{HotReloadMethods.__name__}] {class_name} is not a class in module {module.__name__}")
            return None
        return HotReloadMethods._get_methods_from_class(class_obj)

    @staticmethod
    def _get_module_constants(module: ModuleType) -> MethodMap:
        """Extract module-level constants (non-callable, non-class, non-module attributes)."""
        constants: MethodMap = {}
        for name in dir(module):
            if name.startswith('_'):
                continue
            attr = getattr(module, name, None)
            if attr is None:
                continue
            # Skip imports, classes, functions, and modules
            if inspect.isclass(attr) or inspect.isfunction(attr) or inspect.ismodule(attr) or inspect.ismethod(attr):
                continue
            # Skip callable objects
            if callable(attr):
                continue
            # It's a module-level constant
            constants[name] = MethodInfo(MethodType.CONSTANT, attr)
        return constants

    @staticmethod
    def _update_module_constants(module: ModuleType, constants_to_update: Dict[str, MethodInfo]) -> None:
        """Update module-level constants."""
        for name, info in constants_to_update.items():
            # print(f"[{HotReloadMethods.__name__}] {module.__name__} Update module constant: {name}")
            setattr(module, name, info.func)

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
    def _remove_methods(target_class: Type[Any], deleted_methods: Dict[str, MethodInfo]) -> None:
        """Remove methods and constants that no longer exist in the updated code."""
        for name, info in deleted_methods.items():
            if hasattr(target_class, name):
                print(f"[{HotReloadMethods.__name__}] {target_class.__name__} Remove {info.type.name} {'constant' if info.type == MethodType.CONSTANT else 'method'}: {name}")
                delattr(target_class, name)

    @staticmethod
    def _update_methods(target_class: Type[Any], methods_to_update: Dict[str, MethodInfo]) -> None:
        """Update methods and constants that have changed."""
        for name, info in methods_to_update.items():
            print(f"[{HotReloadMethods.__name__}] {target_class.__name__} Patch {info.type.name} {'constant' if info.type == MethodType.CONSTANT else 'method'}: {name}")
            if info.type == MethodType.STATIC:
                setattr(target_class, name, staticmethod(info.func))
            elif info.type == MethodType.CLASS:
                setattr(target_class, name, classmethod(info.func))
            elif info.type == MethodType.CONSTANT:
                setattr(target_class, name, info.func)
            else:
                setattr(target_class, name, info.func)

    @staticmethod
    def _add_methods(target_class: Type[Any], methods_to_add: Dict[str, MethodInfo]) -> None:
        """Add methods and constants that are new in the updated code."""
        for name, info in methods_to_add.items():
            print(f"[{HotReloadMethods.__name__}] {target_class.__name__} Add {info.type.name} {'constant' if info.type == MethodType.CONSTANT else 'method'}: {name}")
            if info.type == MethodType.STATIC:
                setattr(target_class, name, staticmethod(info.func))
            elif info.type == MethodType.CLASS:
                setattr(target_class, name, classmethod(info.func))
            elif info.type == MethodType.CONSTANT:
                setattr(target_class, name, info.func)
            else:
                setattr(target_class, name, info.func)

    class FileChangeHandler(FileSystemEventHandler):
        def __init__(self, reloader: 'HotReloadMethods') -> None:
            self._reloader: HotReloadMethods = reloader
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
                self._reloader.on_file_modified()
