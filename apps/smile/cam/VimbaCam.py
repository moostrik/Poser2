from apps.smile.cam.vimba import *
from apps.smile.cam.settings import CameraSettings
from apps.smile.cam.FeatureWriter import FeatureWriter
import cv2
import numpy as np
from dataclasses import dataclass
from threading import Thread, Lock, Event
from time import sleep, monotonic
from typing import Optional
import copy
import logging

logger = logging.getLogger(__name__)

FRAME_TIMEOUT: float = 5.0      # s without a complete frame while capturing: the camera is opened again
RETRY_INTERVAL: float = 5.0     # s between two tries to find and open the camera
STREAM_PAUSE: float = 1.0       # s after a stream that ended, before the camera is opened again
GRAB_LOG_INTERVAL: float = 10.0 # s between the log lines of a frame grab that keeps failing
SMALL_PACKET: int = 1500        # bytes: the packet size without jumbo frames on the path


@dataclass
class CameraSetup:
    """What the camera was set up with at open: the region of interest its user set holds and the
    packet size the path accepted."""
    packet_size: int
    width: int
    height: int
    offset_x: int
    offset_y: int

    @property
    def region(self) -> str:
        return f'{self.width}x{self.height} at {self.offset_x},{self.offset_y}'


class NoCamera(Exception):
    """The camera was not found or could not be set up; the message says why."""


def get_camera(camera_id: Optional[str]) -> Camera:
    with Vimba.get_instance() as vimba:
        cams = vimba.get_all_cameras()
        if not cams:
            raise NoCamera('no camera found')

        if camera_id:
            try:
                return vimba.get_camera_by_id(camera_id)

            except VimbaCameraError:
                logger.warning('Vimba camera %s not found, taking the first', camera_id)

        return cams[0]


def setup_camera(camera: Camera, user_set: Optional[str]) -> CameraSetup:
    """Load the user set and let the camera take the largest packet the path accepts; returns that
    packet size with the region of interest the set holds. Raises NoCamera when the camera can't be
    opened or set up."""
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
                        raise NoCamera(f'camera [{camera.get_id()}] failed to load {user_set} from flash')

                # after the user set: it holds a packet size of its own and would put that back
                try:
                    camera.GVSPAdjustPacketSize.run() # type: ignore

                    while not camera.GVSPAdjustPacketSize.is_done(): # type: ignore
                        pass

                except (AttributeError, VimbaFeatureError):
                    raise NoCamera(f'camera [{camera.get_id()}] could not set its packet size')

                def read(name: str) -> int:
                    return int(camera.get_feature_by_name(name).get())

                return CameraSetup(read('GevSCPSPacketSize'), read('Width'), read('Height'), read('OffsetX'), read('OffsetY'))
        except NoCamera:
            raise
        except Exception as e:
            raise NoCamera(f'could not access the camera (already in use?): {e}')


class VimbaCam(Thread):
    def __init__(self, settings: CameraSettings) -> None:
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

        # the last frame as the camera sent it, rotated and flipped; Camera fits it to the render
        self.outputFrame: np.ndarray = np.zeros((1, 1, 3), np.uint8)

        self.rotate90: bool         = False

        # gain, white balance and colour: sent to the camera when the settings change, checked every frame
        self._features: FeatureWriter = FeatureWriter(settings)

        self.capturing: bool = False
        self.camera: Camera | None = None       # None while it is not found: the thread keeps looking

        self._running: bool = False
        self._stop_event: Event = Event()       # (Thread has a _stop of its own)
        self._problem: str = ''                 # why there are no frames, as logged; '' while they come
        self._packet_size: int = 0
        self._grab_error_time: float = 0.0

    def run(self) -> None:
        # The thread streams, and starts over from the search when the camera was not found or its
        # frames stopped (a cable, its power, a lost heartbeat): open() only makes the first try
        while self._running:
            camera: Camera | None = self.camera
            if camera is None:
                camera = self._find()
                if camera is None:
                    self._stop_event.wait(RETRY_INTERVAL)
                    continue
                self.camera = camera
            if not self.capturing:
                sleep(0.01)
                continue
            try:
                self._stream(camera)
            except Exception as e:
                self._warn(f'the stream ended: {e}')
            if self._running and self.capturing:
                self.camera = None
                self._stop_event.wait(STREAM_PAUSE)

    def _stream(self, camera: Camera) -> None:
        """Stream until the capture stops, or until no complete frame arrived for FRAME_TIMEOUT."""
        with Vimba.get_instance():
            with camera:
                camera.start_streaming(handler=self.frameCallback, buffer_count=5, allocation_mode = AllocationMode.AllocAndAnnounceFrame)
                try:
                    last_frame: float = monotonic()
                    while self.capturing and self._running:
                        newFrame = self._take_internal_frame()
                        if newFrame is not None:
                            last_frame = monotonic()
                            if self._problem:
                                logger.info('Vimba camera delivers frames again (packet size %d)', self._packet_size)
                                self._problem = ''
                            if self.rotate90:
                                newFrame = cv2.transpose(newFrame)
                            if self.settings.flip_v:
                                newFrame = cv2.flip(newFrame, 0)
                            if self.settings.flip_h:
                                newFrame = cv2.flip(newFrame, 1)

                            for c in self.frameCallbacks:
                                c(newFrame)

                            with self.frameMutex:
                                self.outputFrame = newFrame
                                self.frameNew = True
                        elif monotonic() - last_frame > FRAME_TIMEOUT:
                            self._warn(f'no complete frame for {FRAME_TIMEOUT:.0f} s, opening the camera again')
                            return

                        # also without frames: the readouts then show that none arrive
                        self._features.update(camera)

                        sleep(0.01)
                finally:
                    camera.stop_streaming()

    def _find(self) -> Camera | None:
        """Find the camera and set it up (user set, packet size); None when that does not work, with
        the reason in the log once."""
        try:
            camera: Camera = get_camera(self.camID)
            setup: CameraSetup = setup_camera(camera, self.user_set)
        except NoCamera as e:
            self._warn(str(e))
            return None
        except Exception as e:
            self._warn(f'could not open the camera: {e}')
            return None
        if not self._problem:
            logger.info('Vimba camera [%s]: loaded %s, %s, packet size %d', camera.get_id(), self.user_set, setup.region, setup.packet_size)
        if setup.packet_size <= SMALL_PACKET:
            logger.warning('Vimba camera packet size is %d: Jumbo Packet is off on the adapter or a switch (see the README), frames drop more easily',
                           setup.packet_size)
        self._packet_size = setup.packet_size
        # the user set was loaded: the panel's values go to the camera again
        self._features.reset()
        return camera

    def _warn(self, problem: str) -> None:
        """A reason there are no frames: logged when it is a new one, not at every retry."""
        if problem != self._problem:
            self._problem = problem
            logger.warning('Vimba camera: %s', problem)

    def __exit__(self) -> None:
        self.close()

    def open(self) -> bool:
        """The first try, and the start of the thread that keeps trying. True when the camera is there."""
        if self.is_alive():
            return self.camera is not None

        self.camera = self._find()
        self._running = True
        self.start()
        return self.camera is not None


    def close(self) -> None:
        self._features.close()

        if not self.is_alive():
            return

        self.stopCapture()
        self._running = False
        self._stop_event.set()
        self.join()
        self.camera = None


    def startCapture(self) -> None:
        # also without a camera: the stream starts when it is found
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
            if monotonic() - self._grab_error_time > GRAB_LOG_INTERVAL:
                self._grab_error_time = monotonic()
                logger.warning('Vimba camera: failed to grab a frame: %s', e)
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