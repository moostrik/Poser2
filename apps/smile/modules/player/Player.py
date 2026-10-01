import logging
import os
import random
import threading
import time

import numpy as np

from modules.player.ImgLoader import LoadImages
from modules.player.Render import PlayerRender
from modules.player.ClipPool import Clip, ClipPool, next_start
from modules.player.FolderScanner import FolderScanner

from modules.osc.Sender import Sender as Osc
from modules.osc.settings import OscSettings
from modules.gl import FullscreenMode, MonitorId, WindowSettings

logger = logging.getLogger(__name__)

STATS_INTERVAL: float = 10.0    # s between the log lines with the frame rates
START_DELAY: float = 0.5        # s between two cells starting when the loaded clips go round again
MIN_START_GAP: float = 0.1      # s at least between two cells starting, when clips load faster than that


def monitor_count() -> int:
    """Connected monitors, as the window manager will count them."""
    import glfw
    return len(glfw.get_monitors()) if glfw.init() else 1


def place_window(monitor: int, windowed: bool, monitors: int) -> tuple[int, bool]:
    """The monitor and mode to open on. Without its own monitor a player takes a plain window on the
    primary one: fullscreen there would cover SMILE."""
    if monitor >= monitors:
        return 0, True
    return monitor, windowed


class Player():
    def __init__(self, osc_port: int, root: str = 'output/', monitor: int = 1, trail: float = 0.9,
                 clips: int = 24, pool: int = 30, hwdecode: int | None = None, windowed: bool = False,
                 fps: float = 30.0, spread: float = 0.0, follow_style: bool = True, recent: int = 3) -> None:
        osc_settings = OscSettings()
        osc_settings.port = int(osc_port)
        self.osc:Osc = Osc(osc_settings)

        self._running: bool = False

        ids: list[str] = [str(i) for i in range(clips)]

        placed: tuple[int, bool] = place_window(monitor, windowed, monitor_count())
        if placed != (monitor, windowed):
            logger.warning('Monitor %d is not connected: the player runs in a window on the primary monitor', monitor)
            monitor, windowed = placed

        # borderless fullscreen: exclusive fullscreen minimizes when another player or SMILE gets focus
        window = WindowSettings()
        window.title = f'player {monitor}'
        window.monitor = MonitorId(monitor)
        window.width = 1920
        window.height = 1024
        window.secondary_list = []
        # a plain window for testing (Esc quits, f toggles borderless fullscreen), else borderless fullscreen
        window.fullscreen_mode = FullscreenMode.WINDOWED if windowed else FullscreenMode.WINDOWED_FULLSCREEN
        window.initialize()
        self.render: PlayerRender = PlayerRender(window, ids, trail)
        self.render.add_exit_callback(self.stop)

        # the clips are decoded once into RAM at the size the cells show them (the portrait is turned a quarter)
        cell_size: tuple[int, int] = (PlayerRender.CELL_HEIGHT, PlayerRender.CELL_WIDTH)
        self.pool: ClipPool = ClipPool(pool, cell_size, hwdecode, recent)
        self.scanner: FolderScanner = FolderScanner(os.path.join(root, 'grins'), self.pool, pool, follow_style, recent)

        self.loaders: list[LoadImages] = []
        for id in ids:
            loader = LoadImages(id, self.pool, fps, spread)
            loader.add_image_callback(self.render.set_image)
            loader.add_image_callback(self._count_frame)
            loader.add_loader_callback(self.send_osc)
            self.loaders.append(loader)

        # a log line every STATS_INTERVAL: frames delivered per cell, render fps, clips in the pool
        self._frames: int = 0
        self._frames_lock = threading.Lock()
        self._stats_stop = threading.Event()
        self._stats_thread = threading.Thread(target=self._stats_loop, name='PlayerStats', daemon=True)
        # the cells start one by one, each on the next clip that is loaded
        self._start_thread = threading.Thread(target=self._start_loop, name='PlayerStart', daemon=True)
        logger.info('Player %d: %d cells, pool %d (newest %d always shown), %s, %s, root %s, %s', monitor, clips, pool, recent,
                    'following the style' if follow_style else 'all styles',
                    (f'{fps:g} fps' if fps > 0 else 'clip fps') + (f' +-{spread:.0%}' if spread > 0 else ''), root, 'window' if windowed else 'fullscreen')

    def start(self) -> None:
        self.pool.start()
        self.scanner.start()
        for l in self.loaders:
            l.start()
        self.osc.start()
        self.render.start()
        self._stats_thread.start()
        self._start_thread.start()
        self._running = True

    def _start_loop(self) -> None:
        """Give every cell its first clip: each clip as it is loaded, in a random cell. With fewer
        clips than cells, go round the loaded clips again, START_DELAY apart."""
        empty: list[LoadImages] = list(self.loaders)
        random.shuffle(empty)
        started: set[str] = set()
        last_start: float = 0.0
        while empty and not self._stats_stop.is_set():
            clip: Clip | None = next_start(self.pool.clips(), started)
            if clip is not None and (not started or time.time() - last_start >= MIN_START_GAP):
                self.pool.use(clip)
                empty.pop().assign(clip)
                started.add(clip.id)
                last_start = time.time()
                logger.info('Cell %d of %d starts on %s', len(self.loaders) - len(empty), len(self.loaders), clip.id)
                continue
            if clip is None and started and self.pool.idle and time.time() - last_start >= START_DELAY:
                started.clear()     # every loaded clip has started a cell: round again from the first
                continue
            self._stats_stop.wait(0.02)

    def stop(self) -> None:
        self._stats_stop.set()
        self.osc.stop()
        self.render.stop()
        for l in self.loaders:
            l.stop()
        self.scanner.stop()
        self.pool.stop()
        self._running = False

    def _count_frame(self, id: str, image: np.ndarray, initial: bool) -> None:
        with self._frames_lock:
            self._frames += 1

    def _stats_loop(self) -> None:
        window = self.render.window_manager.settings
        while not self._stats_stop.wait(STATS_INTERVAL):
            with self._frames_lock:
                frames, self._frames = self._frames, 0
            shown, most = self.pool.spread()
            logger.info('%.1f frames/s per cell, render %d fps (min %d), %d clips on %d cells: %d shown, most %dx',
                        frames / STATS_INTERVAL / len(self.loaders), window.avg_fps, window.min_fps,
                        len(self.pool), len(self.loaders), shown, most)

    def isRunning(self) -> bool :
        return self._running

    def send_osc(self, panning: int) -> None:
        self.osc.send('/bang', panning)
