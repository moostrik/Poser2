"""The clip player of the sphere: one pool of clips for the 48 cells of the two walls, each cell playing
on its own thread, the walls composited by `WallRender` into SMILE's secondary windows. No window and no
OSC in here but the `/bang` to the Max patch; SMILE's laughter test and scrolling name call `light` and
`scroll_loop` directly."""

import logging
import os
import random
import threading
import time
from typing import Any

import numpy as np

from apps.smile import sphere
from apps.smile.credit import CreditLines
from apps.smile.player.clip_loader import ClipLoader
from apps.smile.player.wall_render import WallRender
from apps.smile.player.clip_pool import Clip, ClipPool, next_start
from apps.smile.player.folder_scanner import FolderScanner

from modules.inout import OscSender, OscSenderSettings

logger = logging.getLogger(__name__)

STATS_INTERVAL: float = 10.0    # s between the log lines with the frame rates
START_DELAY: float = 0.5        # s between two cells starting when the loaded clips go round again
MIN_START_GAP: float = 0.1      # s at least between two cells starting, when clips load faster than that
BANG_PORT: int = 4115           # the Max patch's `udpreceive` for wall 0's `/bang`; wall 1 is the next port
CELLS: int = sphere.CELLS       # per wall


def online_cell_mode(active: bool, connected: bool, showing: bool) -> str:
    """What the online cell shows: a clip like any cell while SMILE Online is off or the relay is not there
    (`connected`: with its pass for the code's url), the QR card to the phone page while it waits for someone,
    the main screen during a turn."""
    if not active or not connected:
        return 'clip'
    return 'screen' if showing else 'qr'


class Player:
    def __init__(self, settings: Any, walls: list[WallRender], render_settings: Any, root: str = 'apps/smile/output/') -> None:
        self.settings: Any = settings           # WallsSettings
        self.walls: list[WallRender] = walls
        self._name: Any = render_settings       # its credit_name goes over the captions
        # the Max patch places a clip's laugh by the cell its play starts in: `/bang <cell>` to the wall's port
        self.osc: list[OscSender] = []
        for wall in range(len(walls)):
            osc_settings = OscSenderSettings()
            osc_settings.port = BANG_PORT + wall
            self.osc.append(OscSender(osc_settings, events=('/bang',)))
        self._lines: CreditLines = CreditLines(str(render_settings.credit_lines))     # the captions' lines, one at random per play
        self._lines_lock = threading.Lock()
        # the cell (0 to 47) that is SMILE Online's: no bang and no caption for the clip playing on unseen
        self.online_cell: int | None = None

        self._running: bool = False

        ids: list[str] = [str(i) for i in range(len(walls) * CELLS)]

        # the clips are decoded once into RAM at the size the cells show them (the portrait is turned a quarter)
        cell_size: tuple[int, int] = (WallRender.CELL_HEIGHT, WallRender.CELL_WIDTH)
        hwaccel: int | None = int(settings.hwaccel) if int(settings.hwaccel) >= 0 else None
        pool_size: int = int(settings.pool)
        recent: int = int(settings.recent)
        self.pool: ClipPool = ClipPool(pool_size, cell_size, hwaccel, recent)
        self.scanner: FolderScanner = FolderScanner(os.path.join(root, 'grins'), self.pool, pool_size, recent, float(settings.recycle_every))

        self.loaders: list[ClipLoader] = []
        for id in ids:
            wall, cell = divmod(int(id), CELLS)
            loader = ClipLoader(id, self.pool, settings, sphere.sector_of_cell(wall, cell))
            loader.add_image_callback(self._on_image)
            loader.add_loader_callback(self._on_play)
            self.loaders.append(loader)

        # a log line every STATS_INTERVAL: frames delivered per cell, clips in the pool
        self._frames: int = 0
        self._frames_lock = threading.Lock()
        self._stats_stop = threading.Event()
        self._stats_thread = threading.Thread(target=self._stats_loop, name='PlayerStats', daemon=True)
        # the cells start one by one, each on the next clip that is loaded
        self._start_thread = threading.Thread(target=self._start_loop, name='PlayerStart', daemon=True)
        logger.info('Player: %d cells on %d walls, pool %d (last %d always on every third of the sphere), %s, root %s, %s',
                    len(ids), len(walls), pool_size, recent,
                    f'a clip from disk after {float(settings.recycle_every):g} s without a new one' if float(settings.recycle_every) > 0 else 'no recycling',
                    root, f'GPU decode on adapter {hwaccel}' if hwaccel is not None else 'software decode')

    def start(self) -> None:
        self.pool.start()
        self.scanner.start()
        for l in self.loaders:
            l.start()
        for osc in self.osc:
            osc.start()
        self._stats_thread.start()
        self._start_thread.start()
        self._running = True

    def _start_loop(self) -> None:
        """Give every cell its first clip: each clip as it is loaded, in a random cell. With fewer
        clips than cells, go round the loaded clips again, START_DELAY apart."""
        empty: list[ClipLoader] = list(self.loaders)
        random.shuffle(empty)
        started: set[str] = set()
        last_start: float = 0.0
        while empty and not self._stats_stop.is_set():
            clip: Clip | None = next_start(self.pool.clips(), started)
            if clip is not None and (not started or time.time() - last_start >= MIN_START_GAP):
                loader: ClipLoader = empty.pop()
                self.pool.use(clip, loader.sector)
                loader.assign(clip)
                started.add(clip.id)
                last_start = time.time()
                logger.info('Cell %d of %d starts on %s', len(self.loaders) - len(empty), len(self.loaders), clip.id)
                continue
            if clip is None and started and self.pool.idle and time.time() - last_start >= START_DELAY:
                started.clear()     # every loaded clip has started a cell: round again from the first
                continue
            self._stats_stop.wait(0.02)

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._stats_stop.set()
        for osc in self.osc:
            osc.stop()
        for l in self.loaders:
            l.stop()
        self.scanner.stop()
        self.pool.stop()

    def _stats_loop(self) -> None:
        while not self._stats_stop.wait(STATS_INTERVAL):
            with self._frames_lock:
                frames, self._frames = self._frames, 0
            shown, most = self.pool.spread()
            logger.info('%.1f frames/s per cell, %d clips on %d cells: %d shown, most %dx',
                        frames / STATS_INTERVAL / len(self.loaders), len(self.pool), len(self.loaders), shown, most)

    def is_running(self) -> bool:
        return self._running

    # THE WALLS: a cell's wall and its cell there
    @staticmethod
    def _place(cell: int) -> tuple[int, int]:
        return divmod(cell, CELLS)

    def light(self, wall: int, cell: int, seconds: float, number: int = 0) -> None:
        """The laughter test: the cell white with the screen's `number` for `seconds`."""
        self.walls[wall].light(str(cell), seconds, number)

    def scroll_loop(self, wall: int, word: str, speed: float, gap: float, size: int, ring: int, started: float,
                    cells: list[tuple[int, int]]) -> None:
        """The scrolling name: a word going round a ring, with the wall's cells on it and their places."""
        if self.walls[wall].scroll_loop(word, speed, gap, size, ring, started, [(str(cell), place) for cell, place in cells]):
            logger.info('Scrolling name %s going round over wall %d cells %s', word, wall + 1, ', '.join(f'{cell}@{place}' for cell, place in cells))

    # THE LOADERS' THREADS
    def _on_image(self, id: str, image: np.ndarray, initial: bool) -> None:
        wall, cell = self._place(int(id))
        self.walls[wall].set_image(str(cell), image, initial)
        with self._frames_lock:
            self._frames += 1

    def _on_play(self, cell: int) -> None:
        """A clip starts in a cell: the Max patch places its laugh by that screen's angle
        (data/audio/screen_angles.txt, by cell for port 4115 and cell + 24 for 4116), and the cell gets a
        caption for this play, the artist's name and a random line, as on the main screen. The lines are
        not tied to the clips. Nothing for the online cell: its clip plays unseen."""
        if cell == self.online_cell:
            return
        wall, index = self._place(cell)
        self.osc[wall].submit('/bang', index)
        with self._lines_lock:
            line: str = self._lines.pick()
        self.walls[wall].set_caption(str(index), str(self._name.credit_name), line)
