from modules.gl.MultiWindowRender import MultiWindowRender
from modules.Settings import Settings


class TestMultiWindow():
    def __init__(self, settings: Settings) -> None:
        self.settings: Settings = settings

        self.multi_window_render = MultiWindowRender(settings, self.stop)

        self.running: bool = False

    def start(self) -> None:
        self.multi_window_render.start()
        self.running = True

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False

        self.multi_window_render.stop()

    def isRunning(self) -> bool:
        return self.running
