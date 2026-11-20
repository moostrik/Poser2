import time
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver


class FileModifiedHandler(FileSystemEventHandler):
    """Calls `callback(path)` when a file in the watched folder is modified, at most once per 0.5 s."""

    def __init__(self, callback: Callable[[str], None]) -> None:
        self.callback: Callable[[str], None] = callback
        self.last_modified: float = time.perf_counter()

    def on_modified(self, event: FileSystemEvent) -> None:
        if time.perf_counter() - self.last_modified < 0.5:
            return
        self.last_modified = time.perf_counter()
        if not event.is_directory:
            self.callback(str(event.src_path))


def monitor_path(path: str, callback: Callable[[str], None]) -> BaseObserver:
    """A started observer of the folder `path` (not its subfolders) that calls `callback` with each modified file."""
    event_handler = FileModifiedHandler(callback)
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=False)
    observer.start()
    return observer
