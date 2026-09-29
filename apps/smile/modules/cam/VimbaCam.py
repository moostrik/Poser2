from modules.cam.vimba import *
from modules.cam.settings import CameraSettings
import cv2
import numpy as np
from threading import Thread, Lock, Event
from time import sleep
from typing import Optional
import copy

def get_camera(camera_id: Optional[str]) -> Camera | None:
    with Vimba.get_instance() as vimba:
        cams = vimba.get_all_cameras()
        if not cams:
            print('Vimba', 'no cameras found')
            return None

        if camera_id:
            try:
                return vimba.get_camera_by_id(camera_id)

            except VimbaCameraError:
                print('Vimba cam', camera_id, 'not found defaulting to first')

        # print (cams[0], 'found')
        return cams[0]


def setup_camera(camera: Camera, user_set: Optional[str]) -> bool:
    with Vimba.get_instance():
        try:
            with camera:
                try:
                    camera.GVSPAdjustPacketSize.run() # type: ignore

                    while not camera.GVSPAdjustPacketSize.is_done(): # type: ignore
                        pass

                except (AttributeError, VimbaFeatureError):
                    print('could not set package size')
                    return False

                if user_set:
                    try:

                        feature = camera.get_feature_by_name('UserSetSelector')
                        feature.set(user_set)
                        cmd = camera.get_feature_by_name('UserSetLoad')
                        cmd.run()

                        while not cmd.is_done():
                            pass

                    except VimbaFeatureError:
                        print('Camera [{}] Failed to load user set \'{}\' from flash.'.format(camera.get_id(), user_set))
                        return False

                    print('Camera [{}] Loaded \'{}\' from flash.'.format(camera.get_id(), user_set))
                    return True

                return True
        except:
            print('could not access camera, already connected?')
            return False


class VimbaCam(Thread):
    def __init__(self, settings: CameraSettings, forceSize: tuple[int, int] | None = None) -> None:
        super().__init__()
        self.settings: CameraSettings = settings

        self.user_set: str = 'UserSet2'

        self._internal_frame: np.ndarray
        self._internal_frame_new: bool = False
        self._internal_frame_mutex: Lock = Lock()

        self.frameMutex: Lock = Lock()
        self.frameNew: bool = False
        self.camID: str | None = None
        self.frameCallbacks: list = []

        self.outWidth: int = 1920
        self.outHeight: int = 1080

        if (forceSize):
            self.outWidth, self.outHeight = forceSize
        self.outputFrame: np.ndarray = np.zeros((self.outWidth, self.outHeight,3)) # type: ignore

        self.rotate90: bool         = False

        # gain values last applied to the camera, compared to the settings every frame
        self._gain_auto_applied: bool | None = None
        self._gain_target_applied: int | None = None

        self.capturing: bool = False
        self.camera: Camera | None = None

    def run(self) -> None:

        while self.camera:
            if self.capturing:
                with Vimba.get_instance():
                    with self.camera:
                        self.camera.start_streaming(handler=self.frameCallback, buffer_count=5, allocation_mode = AllocationMode.AllocAndAnnounceFrame)
                        # print('start streaming')
                        while self.capturing:
                            if self._internal_frame_new:
                                newFrame = self._get_internal_frame()
                                if self.rotate90:
                                    newFrame = cv2.transpose(newFrame)
                                if self.settings.flip_v:
                                    newFrame = cv2.flip(newFrame, 0)
                                if self.settings.flip_h:
                                    newFrame = cv2.flip(newFrame, 1)
                                newFrame = cv2.resize(newFrame, (self.outWidth, self.outHeight))

                                for c in self.frameCallbacks:
                                    c(newFrame)

                                with self.frameMutex:
                                    self.outputFrame = newFrame
                                    self.frameNew = True

                                gain_auto: bool = self.settings.gain_auto
                                if gain_auto != self._gain_auto_applied:
                                    self._gain_auto_applied = gain_auto
                                    feature = self.camera.get_feature_by_name('GainAuto')
                                    if gain_auto:
                                        feature.set('Continuous')
                                    else:
                                        feature.set('Off')

                                gain_target: int = self.settings.gain_target
                                if gain_target != self._gain_target_applied:
                                    self._gain_target_applied = gain_target
                                    feature = self.camera.get_feature_by_name('GainAutoTarget')
                                    feature.set(gain_target)

                            sleep(0.01)
                        if self.camera:
                            self.camera.stop_streaming()
            else:
                sleep(0.01)

    def __exit__(self) -> None:
        self.close()

    def open(self) -> bool:
        # print('open')
        if self.camera: return True

        self.camera = get_camera(self.camID)

        if not self.camera:
            print('Cannot access the webcam.')
            return False
        else:
            if not setup_camera(self.camera, self.user_set):
                print('Cannot setup the webcam.')
                return False

        self.start()
        return True


    def close(self) -> None:

        if not self.is_alive():
            return

        # print('close')
        if self.camera:
            self.stopCapture()
            self.camera = None
        self.join()


    def startCapture(self) -> None:
        # print('capture')
        if self.camera:
            self.capturing = True

    def stopCapture(self) -> None:
        self.capturing = False

    def frameCallback(self, cam: Camera, frame: Frame) -> None:
        if self.camera and self.capturing:
            frame_cpy: Frame = copy.deepcopy(frame)
            self.camera.queue_frame(frame)
            sleep(0.01)

            try:
                cvFrame: np.ndarray = frame_cpy.as_numpy_ndarray()
            except:
                print('failed to grab frame')
                return

            with self._internal_frame_mutex:
                self._internal_frame = cvFrame
                self._internal_frame_new = True



    def _get_internal_frame(self) -> np.ndarray:
        with self._internal_frame_mutex:
            return self._internal_frame

    def isColorFrameNew(self) -> bool:
        value: bool = self.frameNew
        self.frameNew = False
        return value

    def getColorImg(self) -> np.ndarray:
        with self.frameMutex:
            return self.outputFrame

    def iscapturing(self) ->bool:
        return self.capturing

    def isOpen(self) -> bool:
        return self.camera is not None

    def setColorCallback(self, callback) -> None:
        self.frameCallbacks.append(callback)

    def clearColorCallbacks(self) -> None:
        self.frameCallbacks = []

    def setRotate90(self, rotate90: bool) -> None:
        self.rotate90 = rotate90