import cv2
import os
import numpy as np
from threading import Thread, Lock
from time import sleep

from modules.cam.settings import CameraSettings


class Image(Thread):
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

        self.image_folder_path = 'input'
        self.image_files = [f for f in os.listdir(self.image_folder_path) if f.endswith('.jpg')]
        self.load_image()
        self.settings.bind(CameraSettings.image_select, self.load_image)


    def load_image(self, _=None) -> None:
        if not self.image_files:
            print('no .jpg files in', self.image_folder_path)
            return
        index: int = min(max(self.settings.image_select, 0), len(self.image_files) - 1)
        selected_file = self.image_files[index]
        file_path = os.path.join(self.image_folder_path, selected_file)
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            print("Failed to load the image.")
        else:
            # Resize the image to 3840x2160
            self.outputColor = cv2.resize(image, (self.outWidth, self.outHeight)) # type: ignore


    def run(self) -> None:

        while self.deviceOpen:
            if self.capturing:
                for c in self.colorCallbacks: c(self.outputColor)
                sleep(0.04)

            else:
                sleep(0.01)

    def __exit__(self) -> None:
        self.close()

    def open(self) -> bool:
        self.deviceOpen = True
        self.start()
        return True

    def close(self) -> None:
        print('close')
        self.settings.unbind(CameraSettings.image_select, self.load_image)
        if not self.deviceOpen: return
        if self.capturing: self.stopCapture()
        self.deviceOpen = False
        self.join()


    def startCapture(self) -> None:
        print('capture')
        if not self.deviceOpen:
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
