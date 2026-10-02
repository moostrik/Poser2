from modules.cam.vimba import *
from modules.cam.settings import CameraSettings
from modules.cam.FeatureWriter import FeatureWriter
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

                # after the user set: it holds a packet size of its own and would put that back
                try:
                    camera.GVSPAdjustPacketSize.run() # type: ignore

                    while not camera.GVSPAdjustPacketSize.is_done(): # type: ignore
                        pass

                except (AttributeError, VimbaFeatureError):
                    print('could not set package size')
                    return False

                print('Camera [{}] packet size {}'.format(camera.get_id(), camera.get_feature_by_name('GevSCPSPacketSize').get()))
                return True
        except:
            print('could not access camera, already connected?')
            return False


class VimbaCam(Thread):
    def __init__(self, settings: CameraSettings, forceSize: tuple[int, int] | None = None) -> None:
        super().__init__()
        self.settings: CameraSettings = settings

        self.user_set: str = 'UserSet2'

        self._internal_frame: np.ndarray | None = None
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
        self.outputFrame: np.ndarray = np.zeros((self.outHeight, self.outWidth, 3), np.uint8)

        self.rotate90: bool         = False

        # gain, white balance and colour: sent to the camera when the settings change, checked every frame
        self._features: FeatureWriter = FeatureWriter(settings)

        self.capturing: bool = False
        self.camera: Camera | None = None

    def run(self) -> None:

        # close() sets self.camera to None from another thread: the loop works on its own reference
        camera: Camera | None = self.camera
        while self.camera and camera:
            if self.capturing:
                with Vimba.get_instance():
                    with camera:
                        camera.start_streaming(handler=self.frameCallback, buffer_count=5, allocation_mode = AllocationMode.AllocAndAnnounceFrame)
                        # print('start streaming')
                        while self.capturing:
                            newFrame = self._take_internal_frame()
                            if newFrame is not None:
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

                            # also without frames: the readouts then show that none arrive
                            self._features.update(camera)

                            sleep(0.01)
                        camera.stop_streaming()
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
        self._features.close()

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
        # runs on Vimba's thread. The frame's buffer is the driver's and is reused once the frame
        # is queued again, so the pixels are copied first; the frame always goes back
        try:
            if not self.capturing:
                return
            # with packets lost the buffer still holds part of an older image: not shown
            complete: bool = frame.get_status() == FrameStatus.Complete
            self._features.count_frame(complete)
            if not complete:
                return
            cvFrame: np.ndarray = frame.as_numpy_ndarray().copy()
            with self._internal_frame_mutex:
                self._internal_frame = cvFrame
                self._internal_frame_new = True
        except Exception as e:
            print('failed to grab frame', e)
        finally:
            try:
                cam.queue_frame(frame)
            except Exception:
                pass    # the stream is stopping

    def _take_internal_frame(self) -> np.ndarray | None:
        """The newest camera frame, once: None until the next one arrives."""
        with self._internal_frame_mutex:
            if not self._internal_frame_new:
                return None
            self._internal_frame_new = False
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