import logging
import os
import random
import threading

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from modules.player.ClipPool import ClipPool

logger = logging.getLogger(__name__)

CLIP_SUFFIX: str = '.mp4'


def clip_style(path: str) -> str:
    """A clip's style is the folder it was saved in: grins/A/ or grins/B/."""
    return os.path.basename(os.path.dirname(path))


def find_clips(folder: str) -> dict[str, str]:
    """Every clip in the style folders: id -> path. Files directly in the folder have no style and are left out."""
    clips: dict[str, str] = {}
    for style in os.listdir(folder):
        style_dir: str = os.path.join(folder, style)
        if not os.path.isdir(style_dir):
            continue
        for name in os.listdir(style_dir):
            if name.endswith(CLIP_SUFFIX):
                clips[os.path.splitext(name)[0]] = os.path.join(style_dir, name)
    return clips


def list_clips(folder: str, max_files: int) -> dict[str, str]:
    """The newest clips of all styles by name (the name is the time): id -> path."""
    clips: dict[str, str] = find_clips(folder)
    return {id: clips[id] for id in sorted(clips, reverse=True)[:max_files]}


def pick_clips(paths: list[str], style: str, count: int) -> list[str]:
    """A random selection of up to count clips of that style, out of all of them."""
    of_style: list[str] = [path for path in paths if clip_style(path) == style]
    return random.sample(of_style, min(max(count, 0), len(of_style)))


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
    """Watches output/grins/ (a folder per style) and decides which clips the pool loads.

    Default: the newest clips of all styles. With `follow_style`: clips of one style, the style of
    the newest clip. A clip arriving in the other style's folder is the switch: the pool gets a
    random selection of that style's clips from all on disk, and the new clip itself, which is
    queued last and therefore decoded first."""

    def __init__(self, folder: str, pool: ClipPool, max_files: int, follow_style: bool = False) -> None:
        self.folder: str = folder
        self.pool: ClipPool = pool
        self.max_files: int = max_files
        self.follow_style: bool = follow_style
        self.style: str | None = None
        self._known: dict[str, str] = {}    # default: the clips given to the pool; follow_style: every clip seen on disk
        self._lock = threading.Lock()
        self.observer: BaseObserver = Observer()

    def start(self) -> None:
        os.makedirs(self.folder, exist_ok=True)
        # the observer first, so a clip landing during the listing is not missed
        self.observer.schedule(ClipEventHandler(self), self.folder, recursive=True)
        self.observer.start()
        self.update()

    def stop(self) -> None:
        self.observer.stop()
        self.observer.join()

    def update(self) -> None:
        with self._lock:
            try:
                if self.follow_style:
                    self._update_style(find_clips(self.folder))
                else:
                    self._update_newest(list_clips(self.folder, self.max_files))
            except OSError as e:
                logger.warning('Cannot list %s: %s', self.folder, e)

    def _update_newest(self, current: dict[str, str]) -> None:
        for id in self._known.keys() - current.keys():
            self.pool.remove(id)
        for id in sorted(current.keys() - self._known.keys()):     # newest last: on top of the LIFO queue
            self.pool.add(current[id], clip_style(current[id]))
        self._known = current

    def _update_style(self, current: dict[str, str]) -> None:
        for id in self._known.keys() - current.keys():
            self.pool.remove(id)
        new: list[str] = sorted(current.keys() - self._known.keys())
        if self.style is None and new:
            new = new[-1:]      # at startup: the newest clip on disk sets the style
        for id in new:
            style: str = clip_style(current[id])
            if style == self.style:
                self.pool.add(current[id], style)      # a new visitor in the current style
            else:
                self._switch(style, id, current)
        self._known = current

    def _switch(self, style: str, trigger: str, current: dict[str, str]) -> None:
        """The clip `trigger` is the first of another style: the pool changes over to that style."""
        self.style = style
        self.pool.set_style(style)
        others: list[str] = [path for id, path in current.items() if id != trigger]
        picked: list[str] = pick_clips(others, style, self.max_files - 1)
        logger.info('Style %s: loading %d of its clips', style, len(picked) + 1)
        for path in picked:
            self.pool.add(path, style)
        self.pool.add(current[trigger], style)      # last in: decoded first
