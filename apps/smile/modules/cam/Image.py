import cv2
import os
import numpy as np
from threading import Thread, Lock
from time import sleep


class Image(Thread):
    def __init__(self, forceSize: tuple[int, int] | None = None, doDepth: bool = True, doPerson: bool = True) -> None:
        super().__init__()

        self.colorMutex: Lock = Lock()
        self.colorFrameNew: bool = False
        self.colorID: int = 0
        self.colorCallbacks: list = []

        self.outWidth: int = 1920
        self.outHeight: int = 1080

        if (forceSize):
            self.outWidth, self.outHeight = forceSize
        self.outputColor: cv2.Mat = np.zeros((self.outWidth, self.outHeight,3)) # type: ignore

        # self.outputColor = cv2.imread('images/1.jpg')
        # print(self.outputColor.shape)
        # self.outputColor = cv2.resize(self.outputColor, (self.outWidth, self.outHeight))
        # print(self.outputColor.shape)

        # TRANSFORMATION SETTINGS
        self.flipV: bool            = False
        self.flipH: bool            = False
        self.rotate90: bool         = False

        self.deviceOpen: bool = False
        self.capturing: bool = False

        self.image_folder_path = 'input'
        self.image_files = [f for f in os.listdir(self.image_folder_path) if f.endswith('.jpg')]
        self.num_image_files = len(self.image_files) - 1
        self.select_image = self.num_image_files - 1
        self.load_image()


    def load_image(self) -> None:
        if self.select_image == -1:
            return
        selected_file = self.image_files[self.select_image]
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

    def isDepthFrameNew(self) -> bool:
        return self.isColorFrameNew()

    def getColorImg(self) -> cv2.Mat:
        with self.colorMutex:
            return self.outputColor

    def getDepthImg(self) -> cv2.Mat:
        return self.getColorImg()

    def iscapturing(self) ->bool:
        return self.capturing

    def isOpen(self) -> bool:
        return self.deviceOpen

    def setColorCallback(self, callback) -> None:
        self.colorCallbacks.append(callback)

    def clearColorCallbacks(self) -> None:
        self.colorCallbacks = []

    def setDepthCallback(self, callback) -> None:
        self.colorCallbacks.append(callback)

    def setFlipH(self, flipH: bool) -> None:
        self.flipH = flipH

    def setFlipV(self, flipV: bool) -> None:
        self.flipV = flipV

    def setRotate90(self, rotate90: bool) -> None:
        self.rotate90 = rotate90
