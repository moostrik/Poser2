import cv2
import numpy as np
from threading import Thread, Lock
from time import sleep

from modules.cam.settings import CameraSettings


class WebCam(Thread):
    def __init__(self, settings: CameraSettings, forceSize: tuple[int, int] | None = None) -> None:
        super().__init__()
        self.settings: CameraSettings = settings

        self.colorMutex: Lock = Lock()
        self.colorFrameNew: bool = False
        self.colorID: int = 0
        self.colorCallbacks: list = []

        self.outWidth: int = 1920
        self.outHeight: int = 1080

        if (forceSize):
            self.outWidth, self.outHeight = forceSize
        self.outputColor: cv2.Mat = np.zeros((self.outWidth, self.outHeight,3)) # type: ignore

        self.rotate90: bool         = False

        self.deviceOpen: bool = False
        self.capturing: bool = False
        self.capDevice: cv2.VideoCapture | None = None

    def run(self) -> None:

        while self.deviceOpen:
            if self.capturing and  self.capDevice:
                ret, frame = self.capDevice.read()
                if ret:
                    if self.rotate90:
                        frame = cv2.transpose(frame)
                    if self.settings.flip_v:
                        frame = cv2.flip(frame, 0)
                    if self.settings.flip_h:
                        frame = cv2.flip(frame, 1)
                    frame = cv2.resize(frame, (self.outWidth, self.outHeight))
                    with self.colorMutex:
                        self.outputColor = frame
                        self.colorFrameNew = True
                    for c in self.colorCallbacks: c(frame)
            else:
                sleep(0.01)

    def __exit__(self) -> None:
        self.close()

    def open(self) -> bool:
        print('open')
        if self.deviceOpen: return True
        self.capDevice = cv2.VideoCapture(0)
        if not self.capDevice.isOpened():
            raise ValueError('Cannot access the webcam.')

        self.deviceOpen = True
        self.start()
        return True

    def close(self) -> None:
        print('close')
        if not self.deviceOpen: return
        if self.capturing: self.stopCapture()

        self.deviceOpen = False
        if self.capDevice:
            self.capDevice.release()
        self.capDevice = None
        self.join()


    def startCapture(self) -> None:
        print('capture')
        if not self.deviceOpen:
            print('CamDepthAi:start', 'device is not open')
            return
        self.capturing = True

    def stopCapture(self) -> None:
        self.capturing = False


    def isColorFrameNew(self) -> bool:
        value: bool = self.colorFrameNew
        # self.colorFrameNew = False
        return value

    def getColorImg(self) -> cv2.Mat:
        with self.colorMutex:
            return self.outputColor

    def iscapturing(self) ->bool:
        return self.capturing

    def isOpen(self) -> bool:
        return self.deviceOpen

    def setColorCallback(self, callback) -> None:
        self.colorCallbacks.append(callback)

    def clearColorCallbacks(self) -> None:
        self.colorCallbacks = []

    def setRotate90(self, rotate90: bool) -> None:
        self.rotate90 = rotate90
