from player.ImgLoader import LoadImages
from player.Render import Render
from player.FolderScanner import FolderScanner

from modules.osc.Sender import Sender as Osc
import time
import random

class Player():
    def __init__(self, osc_port: str, images_dir = 'images/', posX: int = 0, posY: int = 0, trail = 0.9, interval = 1.0, wait = 5.0, num_portraits = 10) -> None:
        self.osc:Osc = Osc('127.0.0.1', 4115)

        self._running: bool = False
        self.numImages: int = 24

        ids: list[str] = []
        for i in range(self.numImages):
            id: str = str(i)
            ids.append(id)

        self.render: Render = Render(3840, 2048, 512, 640, ids, True, True, posX, posY, trail= trail)

        self.scanner: FolderScanner = FolderScanner(images_dir, num_portraits)
        # self.scanner.update_folder_dict()  # Initial scan

        self.loaders: list[LoadImages] = []
        for id in ids:
            t: float = time.time() + 1.0 / self.numImages + random.randint(0, self.numImages)
            loader = LoadImages(id, self.scanner, interval, wait)
            loader.add_image_callback(self.render.set_image)
            loader.add_loader_callback(self.send_osc)
            self.loaders.append(loader)
            pass


        self.loaders

    def start(self) -> None:
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
        self._running = False

    def isRunning(self) -> bool :
        return self._running

    def send_osc(self, panning: int) -> None:
        self.osc.send('/bang', panning)