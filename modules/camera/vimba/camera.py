"""One Allied Vision GigE camera (Mako G-125C) through the vendored VimbaPython binding, as a `CameraSource`
(CAMERA.md, Vimba)."""

import logging
import time
from dataclasses import dataclass
from threading import Thread
from typing import Any

import numpy as np

from modules.utils import Broadcast, Waker

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
CONTROL_INTERVAL: float = 0.1   # s between two sends of the settings to the camera while streaming
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


def get_camera(serial: str | None) -> Any:
    """The Vimba camera with the serial, or the first one found; raises NoCamera when there is none."""
    with Vimba.get_instance() as vimba:
        cams = vimba.get_all_cameras()
        if not cams:
            raise NoCamera('no camera found')

        if serial:
            try:
                return vimba.get_camera_by_id(serial)

            except VimbaCameraError:
                logger.warning('Vimba camera %s not found, taking the first', serial)

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
                            time.sleep(0.001)

                    except VimbaFeatureError:
                        raise NoCamera(f'camera [{camera.get_id()}] failed to load {user_set} from flash')

                # after the user set: it holds a packet size of its own and would put that back
                try:
                    camera.GVSPAdjustPacketSize.run()

                    while not camera.GVSPAdjustPacketSize.is_done():
                        time.sleep(0.001)

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
    """A `CameraSource` on a Vimba camera. Every frame carries the settings' `rotate90` and flips as flags. No
    detector: no tracklets."""

    def __init__(self, settings: VimbaCameraSettings, cam_id: int = 0) -> None:
        super().__init__(name='VimbaCamera', daemon=True)
        self.settings: VimbaCameraSettings = settings
        self.cam_id: int = cam_id
        self.rotate90: bool = settings.rotate90     # how the camera is mounted, fixed at start
        self.serial: str | None = None          # the camera's serial; None: the first camera found

        self._last_frame: float = 0.0           # the arrival of the newest complete frame (perf_counter)

        self.frame_callbacks: Broadcast[[CameraFrame]] = Broadcast(name='Vimba frame')

        # frame rate, exposure, white balance and color: sent to the camera when the settings change, checked every frame
        self.control: SensorControl = SensorControl(settings)

        self.camera: Any | None = None          # the Vimba camera; None while it is not found: the thread keeps looking

        self.running: bool = False
        self._waker: Waker = Waker()            # set by stop: ends the thread's waits at once
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
        self.frame_callbacks.add_callback(callback)

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
        self._waker.set()

    def join(self, timeout: float | None = None) -> None:
        if self.ident is not None:      # started
            super().join(timeout)
        if not self.is_alive():
            self._waker.close()
        self.camera = None

    def run(self) -> None:
        # The thread streams, and starts over from the search when the camera was not found or its
        # frames stopped (a cable, its power, a lost heartbeat): start() only makes the first try
        while self.running:
            camera: Any | None = self.camera
            if camera is None:
                camera = self._find()
                if camera is None:
                    self._waker.wait(time.perf_counter() + RETRY_INTERVAL)
                    continue
                self.camera = camera
            try:
                self._stream(camera)
            except Exception as e:
                self._warn(f'the stream ended: {e}')
            if self.running:
                self.camera = None
                self._waker.wait(time.perf_counter() + STREAM_PAUSE)

    def _stream(self, camera: Any) -> None:
        """Stream until the camera is stopped, or until no complete frame arrived for FRAME_TIMEOUT. The frames go
        out from Vimba's thread; this one sends the settings to the camera and watches that frames arrive."""
        with Vimba.get_instance():
            with camera:
                start: float = time.perf_counter()
                camera.start_streaming(handler=self._frame_handler, buffer_count=5, allocation_mode=AllocationMode.AllocAndAnnounceFrame)
                try:
                    while self.running:
                        self._waker.wait(time.perf_counter() + CONTROL_INTERVAL)
                        if not self.running:
                            return
                        last_frame: float = max(self._last_frame, start)
                        if last_frame > start and self._problem:
                            logger.info('Vimba camera delivers frames again (packet size %d)', self._packet_size)
                            self._problem = ''
                        if time.perf_counter() - last_frame > FRAME_TIMEOUT:
                            self._warn(f'no complete frame for {FRAME_TIMEOUT:.0f} s, opening the camera again')
                            return

                        # also without frames: the readouts then show that none arrive
                        self.control.update(camera)
                finally:
                    camera.stop_streaming()

    def _find(self) -> Any | None:
        """Find the camera and set it up (user set, packet size); None when that does not work, with
        the reason in the log once."""
        try:
            camera: Any = get_camera(self.serial)
            setup: CameraSetup = setup_camera(camera, self.settings.user_set or None)
        except NoCamera as e:
            self._warn(str(e))
            return None
        except Exception as e:
            self._warn(f'could not open the camera: {e}')
            return None
        if not self._problem:
            logger.info('Vimba camera [%s]: loaded %s, %s, packet size %d', camera.get_id(), self.settings.user_set or 'no user set',
                        setup.region, setup.packet_size)
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
        # runs on Vimba's thread, which hands the frame over. The frame's buffer is the driver's and is
        # reused once the frame is queued again, so the pixels are copied first; the frame always goes back
        arrival: float = time.perf_counter()
        image: np.ndarray | None = None
        try:
            if self.running:
                # with packets lost the buffer still holds part of an older image: not shown
                complete: bool = frame.get_status() == FrameStatus.Complete
                self.control.count_frame(arrival, complete)
                if complete:
                    image = frame.as_numpy_ndarray().copy()
        except Exception as e:
            self._grab_error(e)
        finally:
            try:
                cam.queue_frame(frame)
            except Exception as e:
                logger.debug('Vimba camera: the frame was not queued again (the stream is stopping?): %s', e)
        if image is None:
            return
        self._last_frame = arrival
        try:
            self._emit(image, arrival)
        except Exception as e:
            self._grab_error(e)

    def _grab_error(self, e: Exception) -> None:
        """A frame that could not be grabbed: logged, at most once per GRAB_LOG_INTERVAL."""
        if time.perf_counter() - self._grab_error_time > GRAB_LOG_INTERVAL:
            self._grab_error_time = time.perf_counter()
            logger.warning('Vimba camera: failed to grab a frame: %s', e)

    def _emit(self, image: np.ndarray, arrival: float) -> None:
        """The frame becomes the contract's: as delivered, its rotate90 and flips as flags, stamped with its arrival."""
        self.frame_callbacks(CameraFrame(cam_id=self.cam_id, kind=FrameType.VIDEO, image=image, timestamp=arrival,
                                         rotate90=self.rotate90, flip_v=self.settings.flip_v,
                                         flip_h=self.settings.flip_h))
