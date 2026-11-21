from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver


class FileModifiedHandler(FileSystemEventHandler):
    """Calls `callback(path)` for each modification of a file in the watched folder; the receiver waits for its files
    to settle, so no event is dropped here."""

    def __init__(self, callback: Callable[[str], None]) -> None:
        self.callback: Callable[[str], None] = callback

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self.callback(str(event.src_path))


def monitor_path(path: str, callback: Callable[[str], None]) -> BaseObserver:
    """A started observer of the folder `path` (not its subfolders) that calls `callback` with each modified file."""
    event_handler = FileModifiedHandler(callback)
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=False)
    observer.start()
    return observer
