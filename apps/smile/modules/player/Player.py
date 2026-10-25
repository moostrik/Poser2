import logging
import os
import random
import threading
import time

import numpy as np

from modules.credit import NAME, URL
from modules.player.ImgLoader import LoadImages
from modules.player.Render import PlayerRender
from modules.player.ClipPool import Clip, ClipPool, next_start
from modules.player.FolderScanner import FolderScanner

from modules.osc.Sender import Sender as Osc
from modules.osc.Receiver import Receiver
from modules.osc.settings import OscSettings
from modules.gl import FullscreenMode, MonitorId, WindowSettings
from modules.log_config import crash_summary

logger = logging.getLogger(__name__)

STATS_INTERVAL: float = 10.0    # s between the log lines with the frame rates
START_DELAY: float = 0.5        # s between two cells starting when the loaded clips go round again
MIN_START_GAP: float = 0.1      # s at least between two cells starting, when clips load faster than that
LIGHT_WARNING_INTERVAL: float = 10.0    # s between the log lines of /light messages that make no sense


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
                 fps: float = 30.0, spread: float = 0.0, recent: int = 4, recycle_every: float = 20.0,
                 listen_port: int = 0, credit: bool = False, credit_name: str = NAME, credit_url: str = URL) -> None:
        # the first line of every log: what happened before this start
        logger.info(crash_summary())
        osc_settings = OscSettings()
        osc_settings.port = int(osc_port)
        self.osc:Osc = Osc(osc_settings)
        # SMILE's laughter test lights a cell with its screen's number: /light <cell> <seconds> <number> (0: no port, no test)
        self.receiver: Receiver | None = Receiver(int(listen_port), {'/light': self._on_light}) if listen_port else None
        self._light_warning_time: float = 0.0

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
        self.render: PlayerRender = PlayerRender(window, ids, trail, credit, credit_name, credit_url)
        self.render.add_exit_callback(self.stop)

        # the clips are decoded once into RAM at the size the cells show them (the portrait is turned a quarter)
        cell_size: tuple[int, int] = (PlayerRender.CELL_HEIGHT, PlayerRender.CELL_WIDTH)
        self.pool: ClipPool = ClipPool(pool, cell_size, hwdecode, recent)
        self.scanner: FolderScanner = FolderScanner(os.path.join(root, 'grins'), self.pool, pool, recent, recycle_every)

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
        logger.info('Player %d: %d cells, pool %d (last %d always shown), %s, %s, root %s, %s, %s', monitor, clips, pool, recent,
                    f'a clip from disk after {recycle_every:g} s without a new one' if recycle_every > 0 else 'no recycling',
                    (f'{fps:g} fps' if fps > 0 else 'clip fps') + (f' +-{spread:.0%}' if spread > 0 else ''), root, 'window' if windowed else 'fullscreen',
                    f'/light on port {listen_port}' if listen_port else 'no /light port')
        logger.info('Credit: %s', f'{credit_name}, {credit_url}, one play per cell' if credit else 'none')

    def start(self) -> None:
        self.pool.start()
        self.scanner.start()
        for l in self.loaders:
            l.start()
        self.osc.start()
        self.render.start()
        self._stats_thread.start()
        self._start_thread.start()
        # last: a /light bangs through the sender, which must run by then
        if self.receiver is not None:
            self.receiver.start()
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
        if self.receiver is not None:
            self.receiver.stop()
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

    def send_osc(self, cell: int) -> None:
        """A clip starts in a cell: the Max patch places its laugh by that screen's angle
        (data/audio/screen_angles.txt, by cell for port 4115 and cell + 24 for 4116)."""
        self.osc.send('/bang', cell)

    def _on_light(self, cell, seconds, number=0) -> None:
        """SMILE's laughter test (the receiver's thread): show the cell white with the screen's `number` for
        `seconds`. SMILE bangs Max for the laugh itself, so the sound must come from the numbered screen."""
        try:
            cell, seconds, number = int(cell), float(seconds), int(number)
            valid: bool = 0 <= cell < len(self.loaders) and seconds > 0.0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            if time.time() - self._light_warning_time > LIGHT_WARNING_INTERVAL:
                self._light_warning_time = time.time()
                logger.warning('/light %r %r ignored: %d cells', cell, seconds, len(self.loaders))
            return
        self.render.light(str(cell), seconds, number)
        logger.info('Laughter test: cell %d shows screen %d for %.1f s', cell, number, seconds)
