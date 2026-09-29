from threading import Thread
import requests
import time


class ProgressMessage():
    def __init__(self, image: str | None, progress: float = 0) -> None:
        self.progress: float = progress
        self.image: str | None = image

def progress_api(webUiUrl: str) -> ProgressMessage:
    try:
        response = requests.get(url=f'{webUiUrl}/sdapi/v1/progress')

    except :
        return ProgressMessage(None, 0.0)
    else:
        progress = response.json()['progress']
        image =  response.json()['current_image']
        if image:
            return ProgressMessage(image, progress)
        return ProgressMessage(None, 0.0)


class Progress(Thread):

    def __init__(self, webuiPath: str, callback) -> None:
        super().__init__()
        self.webUiUrl: str = webuiPath
        self.callback = callback
        self._thread_running: bool = False

    def start(self) -> None:
        self._thread_running = True
        super().start()

    def stop(self) -> None:
        self._thread_running = False

    def run(self) -> None:
        timeout: float = 1.0 / 60.0
        while self._thread_running:
           m = progress_api(self.webUiUrl)
           self.callback(m)
           time.sleep(timeout)

    def is_running(self) -> bool:
        return self._thread_running