"""The Vimba backend: one Allied Vision GigE camera (Mako G-125C) through Allied Vision's VimbaPython
binding (vendored in `vimbapython/`, which needs the Vimba 6 SDK: `VIMBA_HOME`, VimbaC) on the
camera contract. The thread keeps the camera: `start()` makes the first try, and the thread searches
again every `RETRY_INTERVAL` while there is none and opens it again (user set, packet size, the
panel's values) when no complete frame arrived for `FRAME_TIMEOUT`; `connected` follows it. The
frames come on Vimba's own thread, are copied there and handed over, rotated and flipped, on this
thread. Without the SDK the binding cannot load: `start()` is an error in the log and the camera
stays unconnected, so an app that has the image source still runs."""

import logging
import time
from dataclasses import dataclass
from threading import Thread, Lock, Event
from typing import Any

import cv2
import numpy as np

from .. import CameraFrame, CameraReadings, FrameType, CameraFrameCallback, TrackletCallback

from .control import SensorControl
from .settings import VimbaCameraSettings

try:
    from .vimbapython import Vimba, FrameStatus, AllocationMode, VimbaCameraError, VimbaFeatureError
    VIMBA_ERROR: str | None = None      # why the binding did not load; None when it did
except Exception as e:                  # the Vimba SDK is not installed on this machine
    VIMBA_ERROR = str(e)

logger = logging.getLogger(__name__)

FRAME_TIMEOUT: float = 5.0      # s without a complete frame while streaming: the camera is opened again
RETRY_INTERVAL: float = 5.0     # s between two tries to find and open the camera
STREAM_PAUSE: float = 1.0       # s after a stream that ended, before the camera is opened again
GRAB_LOG_INTERVAL: float = 10.0 # s between the log lines of a frame grab that keeps failing
SMALL_PACKET: int = 1500        # bytes: the packet size without jumbo frames on the path
USER_SET: str = 'UserSet2'      # the set in the camera's flash that holds the region of interest


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


def get_camera(camera_id: str | None) -> Any:
    """The Vimba camera with the id, or the first one found; raises NoCamera when there is none."""
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


def setup_camera(camera: Any, user_set: str | None) -> CameraSetup:
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


class Camera(Thread):
    """A `CameraSource` on a Vimba camera. `rotate90` turns every frame (portrait mode, fixed at
    start); the flips come from the settings, per frame. No detector: no tracklets."""

    def __init__(self, settings: VimbaCameraSettings, rotate90: bool = False, id: int = 0) -> None:
        super().__init__(name='VimbaCamera', daemon=True)
        self.settings: VimbaCameraSettings = settings
        self.id: int = id
        self.rotate90: bool = rotate90
        self.camera_id: str | None = None       # None: the first camera found

        # the newest frame as Vimba's thread copied it, until this thread takes it
        self._frame: np.ndarray | None = None
        self._frame_new: bool = False
        self._frame_mutex: Lock = Lock()

        self.frame_callbacks: list[CameraFrameCallback] = []

        # frame rate, exposure, white balance and colour: sent to the camera when the settings change, checked every frame
        self.control: SensorControl = SensorControl(settings)

        self.camera: Any | None = None          # the Vimba camera; None while it is not found: the thread keeps looking

        self.running: bool = False
        self._stop_event: Event = Event()       # (Thread has a _stop of its own)
        self._problem: str = ''                 # why there are no frames, as logged; '' while they come
        self._packet_size: int = 0
        self._grab_error_time: float = 0.0

    @property
    def readings(self) -> CameraReadings:
        return self.settings.readings

    @property
    def connected(self) -> bool:
        return self.camera is not None

    def add_frame_callback(self, callback: CameraFrameCallback) -> None:
        self.frame_callbacks.append(callback)

    def add_tracklet_callback(self, callback: TrackletCallback) -> None:
        pass    # no detector on this camera: nothing to deliver

    # LIFECYCLE
    def start(self) -> None:
        """The first try, and the start of the thread that keeps trying; `connected` says whether
        the camera is there. Without the Vimba SDK: an error, and no thread."""
        if self.running:
            return
        if VIMBA_ERROR is not None:
            logger.error('Vimba camera unavailable (is the Vimba 6 SDK installed? VIMBA_HOME): %s', VIMBA_ERROR)
            return
        self.camera = self._find()
        self.running = True
        super().start()

    def stop(self) -> None:
        self.control.close()
        self.running = False
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self.ident is not None:      # started
            super().join(timeout)
        self.camera = None

    def run(self) -> None:
        # The thread streams, and starts over from the search when the camera was not found or its
        # frames stopped (a cable, its power, a lost heartbeat): start() only makes the first try
        while self.running:
            camera: Any | None = self.camera
            if camera is None:
                camera = self._find()
                if camera is None:
                    self._stop_event.wait(RETRY_INTERVAL)
                    continue
                self.camera = camera
            try:
                self._stream(camera)
            except Exception as e:
                self._warn(f'the stream ended: {e}')
            if self.running:
                self.camera = None
                self._stop_event.wait(STREAM_PAUSE)

    def _stream(self, camera: Any) -> None:
        """Stream until the camera is stopped, or until no complete frame arrived for FRAME_TIMEOUT."""
        with Vimba.get_instance():
            with camera:
                camera.start_streaming(handler=self._frame_handler, buffer_count=5, allocation_mode=AllocationMode.AllocAndAnnounceFrame)
                try:
                    last_frame: float = time.perf_counter()
                    while self.running:
                        frame: np.ndarray | None = self._take_frame()
                        if frame is not None:
                            last_frame = time.perf_counter()
                            if self._problem:
                                logger.info('Vimba camera delivers frames again (packet size %d)', self._packet_size)
                                self._problem = ''
                            self._emit(frame)
                        elif time.perf_counter() - last_frame > FRAME_TIMEOUT:
                            self._warn(f'no complete frame for {FRAME_TIMEOUT:.0f} s, opening the camera again')
                            return

                        # also without frames: the readouts then show that none arrive
                        self.control.update(camera)

                        time.sleep(0.01)
                finally:
                    camera.stop_streaming()

    def _find(self) -> Any | None:
        """Find the camera and set it up (user set, packet size); None when that does not work, with
        the reason in the log once."""
        try:
            camera: Any = get_camera(self.camera_id)
            setup: CameraSetup = setup_camera(camera, USER_SET)
        except NoCamera as e:
            self._warn(str(e))
            return None
        except Exception as e:
            self._warn(f'could not open the camera: {e}')
            return None
        if not self._problem:
            logger.info('Vimba camera [%s]: loaded %s, %s, packet size %d', camera.get_id(), USER_SET, setup.region, setup.packet_size)
        if setup.packet_size <= SMALL_PACKET:
            logger.warning('Vimba camera packet size is %d: Jumbo Packet is off on the adapter or a switch (see the README), frames drop more easily',
                           setup.packet_size)
        self._packet_size = setup.packet_size
        # the user set was loaded: the panel's values go to the camera again
        self.control.reset()
        return camera

    def _warn(self, problem: str) -> None:
        """A reason there are no frames: logged when it is a new one, not at every retry."""
        if problem != self._problem:
            self._problem = problem
            logger.warning('Vimba camera: %s', problem)

    # FRAMES
    def _frame_handler(self, cam: Any, frame: Any) -> None:
        # runs on Vimba's thread. The frame's buffer is the driver's and is reused once the frame
        # is queued again, so the pixels are copied first; the frame always goes back
        try:
            if not self.running:
                return
            # with packets lost the buffer still holds part of an older image: not shown
            complete: bool = frame.get_status() == FrameStatus.Complete
            self.control.count_frame(complete)
            if not complete:
                return
            image: np.ndarray = frame.as_numpy_ndarray().copy()
            with self._frame_mutex:
                self._frame = image
                self._frame_new = True
        except Exception as e:
            if time.perf_counter() - self._grab_error_time > GRAB_LOG_INTERVAL:
                self._grab_error_time = time.perf_counter()
                logger.warning('Vimba camera: failed to grab a frame: %s', e)
        finally:
            try:
                cam.queue_frame(frame)
            except Exception:
                pass    # the stream is stopping

    def _take_frame(self) -> np.ndarray | None:
        """The newest camera frame, once: None until the next one arrives."""
        with self._frame_mutex:
            if not self._frame_new:
                return None
            self._frame_new = False
            return self._frame

    def _emit(self, image: np.ndarray) -> None:
        """The frame becomes the contract's: rotated and flipped, with its host time."""
        if self.rotate90:
            image = cv2.transpose(image)
        if self.settings.flip_v:
            image = cv2.flip(image, 0)
        if self.settings.flip_h:
            image = cv2.flip(image, 1)
        frame = CameraFrame(cam_id=self.id, kind=FrameType.VIDEO, image=image, timestamp=time.perf_counter())
        for callback in self.frame_callbacks:
            callback(frame)
