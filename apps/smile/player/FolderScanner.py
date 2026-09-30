import logging
import os
import threading

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from player.ClipPool import ClipPool

logger = logging.getLogger(__name__)

CLIP_SUFFIX: str = '.mp4'


def list_clips(folder: str, max_files: int) -> dict[str, str]:
    """The newest clips by name: id -> path."""
    names: list[str] = sorted((n for n in os.listdir(folder) if n.endswith(CLIP_SUFFIX)), reverse=True)[:max_files]
    return {os.path.splitext(name)[0]: os.path.join(folder, name) for name in names}


class ClipEventHandler(FileSystemEventHandler):
    def __init__(self, scanner: 'FolderScanner') -> None:
        super().__init__()
        self.scanner: FolderScanner = scanner

    def on_any_event(self, event: FileSystemEvent) -> None:
        # a clip renamed in from temp/ arrives as a created event on Windows
        if event.is_directory or event.event_type not in ('created', 'moved', 'deleted'):
            return
        paths = (str(event.src_path), str(getattr(event, 'dest_path', '') or ''))
        if any(p.endswith(CLIP_SUFFIX) for p in paths):
            self.scanner.update()


class FolderScanner:
    """Watches recordings/grins/ and keeps the pool fed with the newest clips."""

    def __init__(self, folder: str, pool: ClipPool, max_files: int) -> None:
        self.folder: str = folder
        self.pool: ClipPool = pool
        self.max_files: int = max_files
        self._known: dict[str, str] = {}
        self._lock = threading.Lock()
        self.observer: BaseObserver = Observer()

    def start(self) -> None:
        os.makedirs(self.folder, exist_ok=True)
        # the observer first, so a clip landing during the listing is not missed
        self.observer.schedule(ClipEventHandler(self), self.folder, recursive=False)
        self.observer.start()
        self.update()
        self.pool.initial_done()

    def stop(self) -> None:
        self.observer.stop()
        self.observer.join()

    def update(self) -> None:
        with self._lock:
            try:
                current: dict[str, str] = list_clips(self.folder, self.max_files)
            except OSError as e:
                logger.warning('Cannot list %s: %s', self.folder, e)
                return
            for id in self._known.keys() - current.keys():
                self.pool.remove(id)
            for id in sorted(current.keys() - self._known.keys()):     # newest last: on top of the LIFO queue
                self.pool.add(current[id])
            self._known = current
