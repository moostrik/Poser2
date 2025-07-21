from modules.gl.MultiWindowRender import MultiWindowRender
from modules.Settings import Settings


class TestMultiWindow():
    def __init__(self, settings: Settings) -> None:
        self.settings: Settings = settings

        self.multi_window_render = MultiWindowRender(settings, self.stop)

        self.is_running: bool = False
        self.is_finished: bool = False

    def start(self) -> None:
        self.multi_window_render.start()
        self.is_running = True

    def stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False
        self.is_finished = True

        self.multi_window_render.stop()

    def isRunning(self) -> bool:
        return self.is_running
