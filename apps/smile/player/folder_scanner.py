import logging
import os
import random
import threading
import time
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from apps.smile.player.clip_pool import ClipPool

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

    The wall shows one style, that of the newest recording. A new recording of that style joins the
    pool's `recent`. One of the other style is the switch: the pool gets that clip and the newest
    others of its style as `recent`, and a random selection of that style from all on disk as
    `archive`; the switching clip is queued last and therefore decoded first.

    A recording is a file newer than every clip seen so far; an older file that appears (clips
    sorted into a folder by hand) is archive and only waits on disk for its turn.

    Recycling keeps the wall changing: whenever no clip has joined for `recycle_every` seconds, one
    clip of the style that is on disk and not in the pool joins the `archive`, from a shuffled deck
    so that all come round before any repeats."""

    def __init__(self, folder: str, pool: ClipPool, max_files: int, recent: int = 0, recycle_every: float = 0.0) -> None:
        self.folder: str = folder
        self.pool: ClipPool = pool
        self.max_files: int = max_files
        self.recent: int = recent                   # the last recordings: the newest of a style are loaded for sure
        self.recycle_every: float = recycle_every   # s without a clip joining before one comes from disk; 0 = never
        self.style: str | None = None
        self._known: dict[str, str] = {}            # every clip seen on disk
        self._newest_seen: str = ''                 # a file with a newer id (its time) is a recording
        self._last_join: float = 0.0
        self._deck: list[str] = []                  # ids waiting to be recycled in, shuffled
        self._clock: Callable[[], float] = time.monotonic
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._recycle_thread = threading.Thread(target=self._recycle_loop, name='ClipRecycle', daemon=True)
        self.observer: BaseObserver = Observer()

    def start(self) -> None:
        os.makedirs(self.folder, exist_ok=True)
        # the observer first, so a clip landing during the listing is not missed
        self.observer.schedule(ClipEventHandler(self), self.folder, recursive=True)
        self.observer.start()
        self.update()
        if self.recycle_every > 0:
            self._recycle_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._recycle_thread.is_alive():
            self._recycle_thread.join()
        self.observer.stop()
        self.observer.join()

    # FILE EVENTS

    def update(self) -> None:
        with self._lock:
            try:
                current: dict[str, str] = find_clips(self.folder)
            except OSError as e:
                logger.warning('Cannot list %s: %s', self.folder, e)
                return
            for id in self._known.keys() - current.keys():
                self.pool.remove(id)
            new: list[str] = sorted(current.keys() - self._known.keys())
            if self.style is None and new:
                new = new[-1:]      # at startup: the newest clip on disk sets the style
            for id in new:
                if id <= self._newest_seen:
                    continue        # an older file that appeared: archive, it waits on disk for its turn
                self._newest_seen = id
                style: str = clip_style(current[id])
                if style == self.style:
                    self.pool.add(current[id], style, recent=True)     # a new recording in the current style
                    self._last_join = self._clock()
                else:
                    self._switch(style, id, current)
            self._known = current

    def _switch(self, style: str, trigger: str, current: dict[str, str]) -> None:
        """The clip `trigger` is the first of another style: the pool changes over to that style."""
        self.style = style
        self.pool.switch(style)
        self._deck = []
        others: list[str] = [current[id] for id in sorted(current) if id != trigger and clip_style(current[id]) == style]
        # with the trigger, the newest others are the last recordings of this style; the rest is drawn at random
        newest: list[str] = others[len(others) - max(self.recent - 1, 0):]
        picked: list[str] = pick_clips(others[:len(others) - len(newest)], style, self.max_files - 1 - len(newest))
        logger.info('Style %s: loading %d of its clips', style, len(picked) + len(newest) + 1)
        for path in picked:
            self.pool.add(path, style, recent=False)
        for path in newest:
            self.pool.add(path, style, recent=True)
        self.pool.add(current[trigger], style, recent=True)     # last in: decoded first, the newest right after it
        self._last_join = self._clock()

    # RECYCLING

    def _recycle_loop(self) -> None:
        while not self._stop_event.wait(1.0):
            try:
                self.tick()
            except Exception as e:
                logger.warning('Recycling failed: %s', e)

    def tick(self) -> None:
        """Bring one clip in from disk when no clip has joined for `recycle_every` seconds."""
        with self._lock:
            if self.recycle_every <= 0 or self.style is None:
                return
            now: float = self._clock()
            if now - self._last_join < self.recycle_every or not self.pool.idle:
                return
            self._last_join = now       # also when there is nothing to bring in: look again after the same wait
            path: str | None = self._next_old()
            if path is not None:
                logger.debug('Recycling %s', os.path.basename(path))
                self.pool.add(path, self.style, recent=False)

    def _next_old(self) -> str | None:
        """The next clip of the style that is on disk and not in the pool; None when there is none."""
        try:
            current: dict[str, str] = find_clips(self.folder)
        except OSError as e:
            logger.warning('Cannot list %s: %s', self.folder, e)
            return None
        loaded: set[str] = self.pool.ids()
        waiting: dict[str, str] = {id: path for id, path in current.items()
                                   if clip_style(path) == self.style and id not in loaded}
        if not waiting:
            return None
        self._deck = [id for id in self._deck if id in waiting]
        if not self._deck:
            self._deck = random.sample(sorted(waiting), len(waiting))      # all of them, before any repeats
        return waiting[self._deck.pop()]
