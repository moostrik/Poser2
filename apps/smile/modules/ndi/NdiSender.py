from cv2 import Mat, cvtColor, COLOR_BGR2BGRA
import NDIlib as ndi # type: ignore
from threading import Thread, Lock
from queue import Queue, Empty

class NdiSender(Thread):
    def __init__(self, name: str = "Ndi_Image") -> None:
        super().__init__()
        self.name: str = name
        self.running: bool = False
        self.queue: Queue = Queue()
        self.timeout: float = 0.1
        self.mutex: Lock = Lock()
        self.ndiSendRGB = None

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        if not ndi.initialize(): # type: ignore
            print("NDI can not be initialized")
            return

        sendSettings = ndi.SendCreate() # type: ignore
        sendSettings.ndi_name = self.name
        self.ndiSendRGB = ndi.send_create(sendSettings) # type: ignore
        frameRGB = ndi.VideoFrameV2() # type: ignore

        self.running = True
        while self.running:
            try:
                image, timecode, xmlData = self.queue.get(True, self.timeout)
            except Empty:
                continue

            frameRGB.metadata = xmlData
            frameRGB.timecode = timecode
            cv_image: Mat = cvtColor(image, COLOR_BGR2BGRA)
            frameRGB.data = cv_image
            frameRGB.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX # type: ignore

            self.mutex.acquire
            ndi.send_send_video_v2(self.ndiSendRGB, frameRGB) # type: ignore
            self.mutex.release

        ndi.send_destroy(self.ndiSendRGB) # type: ignore
        ndi.destroy() # type: ignore

    def sendImage(self, image: Mat, timecode: int = 0, xmlData: bytes = b'') -> None:
        if not self.running: return
        self.queue.put((image, timecode, xmlData))

    def setNdiAddress(self, name: str) -> None:
        if name == self.name: return
        self.name = name

        if not self.running: return
        sendSettings = ndi.SendCreate() # type: ignore
        sendSettings.ndi_name = self.name # type: ignore
        self.mutex.acquire
        ndi.send_destroy(self.ndiSendRGB) # type: ignore
        self.ndiSendRGB = ndi.send_create(sendSettings) # type: ignore
        self.mutex.release