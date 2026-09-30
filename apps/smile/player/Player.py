import os

from player.ImgLoader import LoadImages
from player.Render import PlayerRender
from player.ClipPool import ClipPool
from player.FolderScanner import FolderScanner

from modules.osc.Sender import Sender as Osc
from modules.osc.settings import OscSettings
from modules.gl import FullscreenMode, MonitorId, WindowSettings

class Player():
    def __init__(self, osc_port: int, root: str = 'recordings/', monitor: int = 1, trail: float = 0.9, wait: float = 5.0,
                 clips: int = 24, pool: int = 24, hwdecode: int | None = None) -> None:
        osc_settings = OscSettings()
        osc_settings.port = int(osc_port)
        self.osc:Osc = Osc(osc_settings)

        self._running: bool = False

        ids: list[str] = [str(i) for i in range(clips)]

        # borderless fullscreen: exclusive fullscreen minimizes when another player or SMILE gets focus
        window = WindowSettings()
        window.title = f'player {monitor}'
        window.monitor = MonitorId(monitor)
        window.width = 1920
        window.height = 1024
        window.secondary_list = []
        window.fullscreen_mode = FullscreenMode.WINDOWED_FULLSCREEN
        window.initialize()
        self.render: PlayerRender = PlayerRender(window, ids, trail)
        self.render.add_exit_callback(self.stop)

        # the clips are decoded once into RAM at the size the cells show them (the portrait is turned a quarter)
        cell_size: tuple[int, int] = (PlayerRender.CELL_HEIGHT, PlayerRender.CELL_WIDTH)
        self.pool: ClipPool = ClipPool(pool, cell_size, hwdecode)
        self.scanner: FolderScanner = FolderScanner(os.path.join(root, 'grins'), self.pool, pool)

        self.loaders: list[LoadImages] = []
        for id in ids:
            loader = LoadImages(id, self.pool, wait)
            loader.add_image_callback(self.render.set_image)
            loader.add_loader_callback(self.send_osc)
            self.loaders.append(loader)

    def start(self) -> None:
        self.pool.start()
        self.scanner.start()
        for l in self.loaders:
            l.start()
        self.osc.start()
        self.render.start()
        self._running = True

    def stop(self) -> None:
        self.osc.stop()
        self.render.stop()
        for l in self.loaders:
            l.stop()
        self.scanner.stop()
        self.pool.stop()
        self._running = False

    def isRunning(self) -> bool :
        return self._running

    def send_osc(self, panning: int) -> None:
        self.osc.send('/bang', panning)
