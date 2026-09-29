from cv2 import Mat, cvtColor, COLOR_BGRA2BGR
import NDIlib as ndi # type: ignore
from threading import Thread, Lock
from queue import Queue
from time import sleep

# for more info on NDILib check
# https://github.com/buresu/ndi-python/blob/master/src/main.cpp

def getNameIndex(name: str, sources) -> int:
    cntr = 0
    for s in sources:
        if name in s.ndi_name:
            return cntr
        cntr += 1
    return -1

class NdiReceiver(Thread):
    def __init__(self, frameCallbacks: list, name: str = "Ndi_Image") -> None:
        super().__init__()
        self.name: str = name
        self.running: bool = False
        self.queue: Queue = Queue()
        self.timeout: float = 0.1
        self.mutex: Lock = Lock()

        self.frameCallbacks: list = frameCallbacks

    def stop(self) -> None:
        if not self.running: return
        self.running = False
        self.join()

    def run(self) -> None:
        if not ndi.initialize(): # type: ignore
            print("NDI can not be initialized")
            return

        ndiFinder = ndi.find_create_v2() # type: ignore

        source = None
        self.running = True
        while not source and self.running:
            ndi.find_wait_for_sources(ndiFinder, 0) # type: ignore
            sources = ndi.find_get_current_sources(ndiFinder) # type: ignore
            index: int = getNameIndex(self.name, sources)
            if index >= 0:
                source = sources[index]
            sleep(0.01)
        self.running = False
        if not source: return

        receiveSettings = ndi.RecvCreateV3() # type: ignore
        receiveSettings.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA # type: ignore

        self.ndiReceiveRGB = ndi.recv_create_v3(receiveSettings) # type: ignore
        ndi.recv_connect(self.ndiReceiveRGB, source) # type: ignore
        ndi.find_destroy(ndiFinder) # type: ignore

        self.running = True
        while self.running:
            t, v, a, _ = ndi.recv_capture_v2(self.ndiReceiveRGB, 0) # type: ignore
            if t == ndi.FRAME_TYPE_VIDEO: # type: ignore
                frame: Mat = cvtColor(v.data, COLOR_BGRA2BGR)
                for c in self.frameCallbacks: c(frame, v.timecode, v.metadata)
                ndi.recv_free_video_v2(self.ndiReceiveRGB, v) # type: ignore
            sleep(0.01)

        ndi.recv_destroy(self.ndiReceiveRGB) # type: ignore
        ndi.destroy() # type: ignore
