"""One OAK-D camera on depthai 2 (https://oak-web.readthedocs.io/), as a `CameraSource`: on its sensor, or fed
recorded clips through its own pipeline."""

import logging
import time
from threading import Thread, Event

import depthai as dai
from numpy import ndarray

from .. import CameraResolution, CameraFrame, RigCameraReadings, CameraTracklet, CameraTracklets, FrameType, CameraFrameCallback, \
    TrackletCallback, ClipFrame, ClipSource
from ..geometry import MountConfig, MountReadout, mode_size, frame_size, delivered_height, frame_summary, lens_report
from ..oak_device import capture_time, smoothed_latency, read_lens_calibration, imu_rotation_to_camera, \
    log_connected_sensors, TRACKING_STATUS, ColorSensorControl
from modules.utils import Broadcast, FPS, Rect

from .clip_input import ClipInput
from .control import MonoSensorControl
from .device import get_device_list
from .imu import IMU_BOARD_ROLL
from .pipeline import build_pipeline, VIDEO_STREAM, TRACKLETS_STREAM, IMU_STREAM, CONTROL_STREAM, CLIPS_STREAM
from .settings import OakCameraSettings

logger = logging.getLogger(__name__)

CLIP_QUEUE_SIZE: int = 1    # the host's clip queue; a send that finds it full drops the oldest (`dropped_clips`)


class Camera(Thread):
    """A `CameraSource` on an OAK-D: its warped frames, the tracklets of its detector and tracker, and its readings;
    with `clips`, the recording's frames go through the device in place of the sensor's, without its IMU and sensor
    controls."""

    def __init__(self, settings: OakCameraSettings, cam_id: int = 0, clips: ClipSource | None = None) -> None:
        super().__init__()
        self.stop_event = Event()
        self.running: bool = False
        self.cam_id: int = cam_id

        # SETTINGS (reactive)
        self.settings: OakCameraSettings = settings

        # FIXED SETTINGS (read from INIT fields once)
        self.device_id: str =           settings.device_id
        self.model_path: str =          settings.model_path
        self.fps: float =               settings.fps
        self.square: bool =             settings.square
        self.color: bool =              settings.color
        self.yolo: bool =               settings.yolo
        self.resolution: CameraResolution = settings.resolution
        self.clip_warp: bool =          settings.clip_warp
        self.clips: ClipSource | None = clips

        self.mount: MountConfig = MountConfig(
            flip_h=settings.flip_h,
            flip_v=settings.flip_v,
            tilt=settings.tilt,
            keystone=settings.keystone,
            fov_h=settings.fov,
            lens_fov=settings.lens_fov,
            lens_center_x=settings.lens_center_x,
            lens_center_y=settings.lens_center_y,
        )
        if settings.rotate90:
            # the flips are baked into the warp, whose columns are the panoramic tracker's bearings: no rotate90 on top
            logger.warning(f'{self.device_id}: rotate90 is not applied on an OAK-D')
        self.frame_height: int = delivered_height(self.color, self.resolution, self.mount.fov_h, self.mount.tilt,
                                                  self.mount.lens_fov, self.mount.lens_center)

        # DEVICE — its pipeline is built in `_open`, its queues in `_setup_queues`; all are closed with it.
        self.pipeline: dai.Pipeline | None = None
        self.device: dai.Device | None = None
        self.queues: list[dai.DataOutputQueue | dai.DataInputQueue] = []
        self.control: ColorSensorControl | MonoSensorControl | None = None     # the camera's sensor's
        self.clip_input: ClipInput | None = None       # the open device's clip queue, with clips

        # FPS
        self.video_fps: FPS = FPS()
        self.tracklet_fps: FPS = FPS()

        # CALLBACKS (run in registration order)
        self.frame_callbacks: Broadcast[[CameraFrame]] = Broadcast(name='OAK-D frame')
        self.tracklet_callbacks: Broadcast[[CameraTracklets]] = Broadcast(name='OAK-D tracklets')

        self.mount_readout: MountReadout = MountReadout(self.settings.readings, IMU_BOARD_ROLL)

    @property
    def readings(self) -> RigCameraReadings:
        return self.settings.readings

    @property
    def connected(self) -> bool:
        return self.device is not None

    def start(self) -> None:
        if self.clips is not None:
            self.clips.add_frame_callback(self._clip_callback)      # once: the player has no way to take it back
        super().start()

    def stop(self) -> None:
        self.running = False
        self.stop_event.set()

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                if not self._open():
                    return
                self.stop_event.wait()
                self._close()
            except Exception:
                logger.exception("Camera error")

    # ── Open and close ─────────────────────────────────────────────────

    def _open(self) -> bool:
        device_list: list[str] = get_device_list(verbose=False)

        if self.device_id not in device_list:
            logger.warning(f'{self.device_id} NOT AVAILABLE in {device_list}')
            return False

        self.pipeline = dai.Pipeline()
        self._setup_pipeline(self.pipeline)
        self._log_frame_geometry()

        try:
            self.device = dai.Device(self.pipeline, dai.DeviceInfo(self.device_id))
        except Exception as e:
            logger.error(f'Could not open device: {e}')
            self.pipeline = None
            return False

        self._setup_queues()

        logger.info(f'{self.device_id} OPEN')
        log_connected_sensors(self.device, self.device_id)
        self._read_mount_calibration()
        self.running = True
        return True

    def _close(self) -> None:
        self.clip_input = None
        if self.control is not None:
            self.control.close()
            self.control = None
        if self.device is not None:
            self.device.close()
            self.device = None
        self.pipeline = None
        for queue in self.queues:
            queue.close()
        self.queues.clear()

        self.frame_callbacks.clear()
        self.tracklet_callbacks.clear()

        logger.info(f'{self.device_id} CLOSED')

    def _setup_pipeline(self, pipeline: dai.Pipeline) -> None:
        build_pipeline(pipeline, self.model_path, self.fps, self.color, self.square, self.yolo, self.resolution,
                       self.mount, frame_height=self.frame_height, clips=self.clips is not None, warp_clips=self.clip_warp)

    def _setup_queues(self) -> None:
        """The device's queues: the frames and tracklets every camera has; with clips the clip input, without the IMU
        (it would report a box on a bench) and the sensor controls (there is no sensor); else the IMU and the sensor
        controls."""
        assert self.device is not None
        self._setup_frame_queues()
        if self.clips is not None:
            clips: dai.DataInputQueue = self.device.getInputQueue(name=CLIPS_STREAM, maxSize=CLIP_QUEUE_SIZE, blocking=False)  # pyright: ignore[reportCallIssue]
            self.queues.append(clips)
            expected: tuple[int, int] = (mode_size(self.color, self.resolution) if self.clip_warp
                                         else frame_size(self.color, self.resolution, self.square, self.frame_height))
            self.clip_input = ClipInput(clips, self.color, expected, self.cam_id)
            return
        self._setup_imu_queue()
        control: dai.DataInputQueue = self.device.getInputQueue(CONTROL_STREAM)
        self.queues.append(control)
        self.control = ColorSensorControl(self.settings.color_sensor, control) if self.color \
            else MonoSensorControl(self.settings.mono_sensor, self.device, control)

    def _setup_frame_queues(self) -> None:
        assert self.device is not None
        video: dai.DataOutputQueue = self.device.getOutputQueue(name=VIDEO_STREAM, maxSize=1, blocking=False)  # pyright: ignore[reportCallIssue]
        video.addCallback(self._video_callback)
        self.queues.append(video)
        if self.yolo:
            tracklets: dai.DataOutputQueue = self.device.getOutputQueue(name=TRACKLETS_STREAM, maxSize=1, blocking=False)  # pyright: ignore[reportCallIssue]
            tracklets.addCallback(self._tracker_callback)
            self.queues.append(tracklets)

    def _setup_imu_queue(self) -> None:
        """Subscribe to the IMU, if the board has one. Board revisions differ, so absence is
        normal and leaves `tilt_measured` / `roll_measured` at NaN."""
        assert self.device is not None
        try:
            sensor: str = self.device.getConnectedIMU()
        except Exception as exc:
            logger.debug(f'{self.device_id} could not query the IMU: {exc}')
            return
        if not sensor or sensor.upper() == 'NONE':
            logger.info(f'{self.device_id} has no IMU — mount orientation will not be measured')
            return
        try:
            imu: dai.DataOutputQueue = self.device.getOutputQueue(name=IMU_STREAM, maxSize=1, blocking=False)  # pyright: ignore[reportCallIssue]
            imu.addCallback(self._imu_callback)
            self.queues.append(imu)
            logger.debug(f'{self.device_id} IMU {sensor} reporting mount orientation')
        except Exception as exc:
            logger.debug(f'{self.device_id} no IMU stream: {exc}')

    def _log_frame_geometry(self) -> None:
        """Log at each open the window this camera's warp delivers and the rows the sensor covers; nothing on the
        keystone path, which keeps the raw frame."""
        if self.mount.keystone != 0.0 and self.mount.tilt == 0.0:
            return
        src: tuple[int, int] = mode_size(self.color, self.resolution)
        out: tuple[int, int] = frame_size(self.color, self.resolution, self.square, self.frame_height)
        logger.info(f'{self.device_id} {frame_summary(src, out, self.mount)}')

    def _read_mount_calibration(self) -> None:
        """One-shot at open: what this unit says about its own lens and its IMU's orientation."""
        assert self.device is not None
        socket = dai.CameraBoardSocket.CAM_A if self.color else dai.CameraBoardSocket.CAM_B
        src: tuple[int, int] = mode_size(self.color, self.resolution)
        lens = read_lens_calibration(self.device, socket, src, self.device_id)
        if lens is not None:
            logger.debug(f'{self.device_id} {lens_report(self.settings.readings, lens, src, self.mount)}')
        self.mount_readout.imu_rotation = imu_rotation_to_camera(self.device, socket, self.device_id)

    # ── The device's callbacks: the hand-over to the contract ──────────

    def _clip_callback(self, clip: ClipFrame) -> None:
        """The player's clip frame of this camera, to the open device's clip input; nothing while it is closed. Only
        the video frames: another kind (depth) would be another input."""
        clip_input: ClipInput | None = self.clip_input
        if not self.running or clip_input is None or clip.cam_id != self.cam_id or clip.kind != FrameType.VIDEO:
            return
        clip_input.send(clip)

    def _video_callback(self, msg: dai.ImgFrame) -> None:
        arrival: float = time.perf_counter()
        timestamp: float = capture_time(msg)
        readings: RigCameraReadings = self.settings.readings
        self.video_fps.processed()
        readings.video_fps = self.video_fps.get_rate_average()
        readings.video_min_fps = self.video_fps.get_rate_minimum()
        readings.video_latency = smoothed_latency(readings.video_latency, msg)
        clip_input: ClipInput | None = self.clip_input
        if clip_input is not None:
            readings.dropped_clips += clip_input.returned(msg.getSequenceNum())
        if self.control is not None:
            self.control.readback(msg)
        self._emit_frame(FrameType.VIDEO, msg.getCvFrame(), timestamp, arrival)

    def _tracker_callback(self, msg: dai.Tracklets) -> None:
        """The output queue's callback: the hand-over, where depthai's tracklets become the contract's, stamped with
        their capture (`capture_time`)."""
        arrival: float = time.perf_counter()
        timestamp: float = capture_time(msg)
        self.tracklet_fps.processed()
        self.settings.readings.tracker_fps = self.tracklet_fps.get_rate_average()
        self.settings.readings.tracker_latency = smoothed_latency(self.settings.readings.tracker_latency, msg)
        tracklets: list[CameraTracklet] = [
            CameraTracklet(cam_id=self.cam_id, id=t.id, status=TRACKING_STATUS[t.status],
                           roi=Rect(t.roi.x, t.roi.y, t.roi.width, t.roi.height), age=t.age, timestamp=timestamp)
            for t in msg.tracklets]
        self.settings.readings.tracklets = len(tracklets)
        self._emit_tracklets(tracklets, timestamp, arrival)

    def _imu_callback(self, msg) -> None:
        self.mount_readout.update([(p.acceleroMeter.x, p.acceleroMeter.y, p.acceleroMeter.z) for p in msg.packets])

    def _emit_frame(self, kind: FrameType, image: ndarray, timestamp: float, arrival: float) -> None:
        """The hand-over: the frame, with its capture time (`capture_time`) and its arrival, goes to every callback;
        none after `stop()`."""
        if not self.running:
            return
        self.frame_callbacks(CameraFrame(cam_id=self.cam_id, kind=kind, image=image, timestamp=timestamp,
                                         arrival=arrival))

    def _emit_tracklets(self, tracklets: list[CameraTracklet], timestamp: float, arrival: float) -> None:
        if not self.running:
            return
        self.tracklet_callbacks(CameraTracklets(cam_id=self.cam_id, timestamp=timestamp, tracklets=tracklets,
                                                arrival=arrival))

    # ── CameraSource ───────────────────────────────────────────────────

    def add_frame_callback(self, callback: CameraFrameCallback) -> None:
        self.frame_callbacks.add_callback(callback)

    def add_tracklet_callback(self, callback: TrackletCallback) -> None:
        self.tracklet_callbacks.add_callback(callback)
