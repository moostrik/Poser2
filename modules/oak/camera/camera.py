# DOCS
# https://oak-web.readthedocs.io/
# https://docs.luxonis.com/software/depthai/examples/depth_post_processing/

import logging
from threading import Thread, Event

import depthai as dai
from cv2 import applyColorMap, COLORMAP_JET
from numpy import ndarray

from modules.utils import FPS

from .definitions import CameraResolution, FrameType, Input, Output, Tracklet, FrameCallback, SyncCallback, TrackerCallback, get_device_list, log_connected_sensors, read_lens_calibration, imu_rotation_to_camera, imu_to_camera, unroll_imu_frame, orientation_from_gravity, IMU_SMOOTHING, mode_size, frame_size, frame_window, frame_coverage, coverage_summary, full_frame_height, lens_field, lens_deviation
from .pipeline import setup_pipeline, get_frame_types, WarpConfig
from .settings import CameraSettings

logger = logging.getLogger(__name__)

class Camera(Thread):
    _id_counter = 0
    _pipeline: dai.Pipeline | None = None

    def __init__(self, core_settings: CameraSettings) -> None:
        super().__init__()
        self.stop_event = Event()
        self.running: bool = False

        # ID
        self.id: int =                  Camera._id_counter
        Camera._id_counter +=             1
        self.id_string: str =           str(self.id)

        # SETTINGS (reactive)
        self.settings: CameraSettings =   core_settings

        # FIXED SETTINGS (read from INIT fields once)
        self.device_id: str =           core_settings.device_id
        self.model_path: str =          core_settings.model_path
        self.fps: float =               core_settings.fps
        self.square: bool =             core_settings.square
        self.do_color: bool =           core_settings.color
        self.do_stereo: bool =          core_settings.stereo
        self.do_yolo: bool =            core_settings.yolo
        self.resolution: CameraResolution = core_settings.resolution
        self.frame_height: int =        core_settings.frame_height
        self.show_stereo: bool =        core_settings.depth.show

        self.mount: WarpConfig = WarpConfig(
            flip_h=core_settings.flip_h,
            flip_v=core_settings.flip_v,
            tilt=core_settings.tilt,
            keystone=core_settings.keystone,
            fov_h=core_settings.fov,
            lens_fov=core_settings.lens_fov,
            lens_centre_x=core_settings.lens_centre_x,
            lens_centre_y=core_settings.lens_centre_y,
        )

        # DAI
        self.device:                    dai.Device
        self.inputs: dict[Input, dai.DataInputQueue] = {}
        self.outputs: dict[Output, dai.DataOutputQueue] = {}
        self.num_tracklets: int =       0

        # FPS
        self.fps_counters: dict[FrameType, FPS] = {}
        self.tps_counter =              FPS(120)

        # FRAME TYPES
        self.frame_types: list[FrameType] = get_frame_types(self.do_color, self.do_stereo, self.show_stereo, False)
        self.frame_types.sort(key=lambda x: x.value)

        # CALLBACKS (run in registration order)
        self.preview_callbacks: list[FrameCallback] = []
        self.frame_callbacks: list[FrameCallback] = []
        self.sync_callbacks: list[SyncCallback] = []
        self.tracker_callbacks: list[TrackerCallback] = []

        # MOUNT READOUT — the gravity vector, smoothed, and how to get it into the camera's frame.
        # A tripod does not move, so a single sample is almost all noise; the average is the
        # signal. State lives here and not on the settings object, which stays pure data.
        self._imu_rotation: list[list[float]] | None = None
        self._gravity: tuple[float, float, float] | None = None

        # PREVIEW
        self.preview_type =             FrameType.VIDEO

        self.cntr: int = 0

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

    def _open(self) -> bool:
        device_list: list[str] = get_device_list(verbose=False)

        if self.device_id not in device_list:
            logger.warning(f'{self.device_id} NOT AVAILABLE in {device_list}')
            return False

        if Camera._pipeline is None:
            Camera._pipeline = dai.Pipeline()
            self._setup_pipeline(Camera._pipeline)
            self._log_frame_geometry()

        try:
            self.device = self._try_device(self.device_id, Camera._pipeline, num_tries=1)
        except Exception as e:
            logger.error(f'Could not open device: {e}')
            return False

        self._setup_queues()

        logger.info(f'{self.device_id} OPEN')
        log_connected_sensors(self.device, self.device_id)
        self._read_mount_calibration()
        self.running = True
        self.settings.connect(self.device, self.inputs, self.do_color)
        return True

    def _setup_pipeline(self, pipeline: dai.Pipeline) -> None:
            setup_pipeline(pipeline, self.model_path, self.fps, self.square, self.do_color, self.do_stereo, self.do_yolo, self.resolution, self.show_stereo, self.mount, simulate=False, frame_height=self.frame_height)

    def _log_frame_geometry(self) -> None:
        """Once per pipeline: the window the warp delivers and where the sensor covers it.

        Pure geometry from the preset — no device needed — logged so the rig check can read
        the horizon row and the arch off the log rather than infer them from the picture.
        Not meaningful on the keystone path, which keeps the raw frame."""
        if self.mount.keystone != 0.0 and self.mount.tilt == 0.0:
            return
        src: tuple[int, int] = mode_size(self.do_color, self.resolution)
        out: tuple[int, int] = frame_size(self.do_color, self.resolution, self.square, self.frame_height)
        window = frame_window(src, out, src[0], self.mount.fov_h, self.mount.tilt,
                              self.mount.lens_fov, self.mount.lens_centre)
        coverage = frame_coverage(src, out, src[0], self.mount.fov_h, self.mount.tilt,
                                  self.mount.flip_h, self.mount.flip_v,
                                  self.mount.lens_fov, self.mount.lens_centre)
        full: int = full_frame_height(src, src[0], self.mount.fov_h, self.mount.tilt,
                                      self.mount.lens_fov, self.mount.lens_centre)
        logger.info(f'frame {out[0]}x{out[1]}: elevation {window.elevation_bottom:+.1f} to '
                    f'{window.elevation_top:+.1f} deg, horizon at row {window.horizon_px:.1f}, '
                    f'{window.focal:.1f} px/rad; '
                    f'{coverage_summary(coverage, out, src[0], self.mount.fov_h, self.mount.flip_h)}; '
                    f'the sensor\'s full reach needs frame_height {full}')

    def _read_mount_calibration(self) -> None:
        """One-shot at open: what this unit says about its own lens and its IMU's orientation."""
        socket = dai.CameraBoardSocket.CAM_A if self.do_color else dai.CameraBoardSocket.CAM_B
        src: tuple[int, int] = mode_size(self.do_color, self.resolution)
        lens = read_lens_calibration(self.device, socket, src, self.device_id)
        if lens is not None:
            focal, cx, cy = lens
            field: float = lens_field(focal, src[0])
            error: float = lens_deviation(focal, cx, cy, src, src[0], self.mount.fov_h,
                                          self.mount.lens_fov, self.mount.lens_centre)
            self.settings.readings.fov_factory = field
            self.settings.readings.lens_error = error
            logger.debug(f'{self.device_id} lens: field {field:.1f} deg across {src[0]} px, '
                         f'centre offset ({cx - (src[0] - 1) / 2.0:+.1f}, {cy - (src[1] - 1) / 2.0:+.1f}) px; '
                         f'{error:.2f} deg off the shared lens')
        self._imu_rotation = imu_rotation_to_camera(self.device, socket, self.device_id)

    def _setup_imu_queue(self) -> None:
        """Subscribe to the IMU, if the board has one. Board revisions differ, so absence is
        normal and leaves `tilt_measured` / `roll_measured` at NaN."""
        try:
            sensor: str = self.device.getConnectedIMU()
        except Exception as exc:
            logger.debug(f'{self.device_id} could not query the IMU: {exc}')
            return
        if not sensor or sensor.upper() == 'NONE':
            logger.info(f'{self.device_id} has no IMU — mount orientation will not be measured')
            return
        try:
            self.outputs[Output.IMU_OUT] = self.device.getOutputQueue(name='imu', maxSize=1, blocking=False)  # type: ignore
            self.outputs[Output.IMU_OUT].addCallback(self._imu_callback)
            logger.debug(f'{self.device_id} IMU {sensor} reporting mount orientation')
        except Exception as exc:                    # the stream is absent in simulation
            logger.debug(f'{self.device_id} no IMU stream: {exc}')

    def _imu_callback(self, msg) -> None:
        for packet in msg.packets:
            raw = packet.acceleroMeter
            vector: tuple[float, float, float] = unroll_imu_frame(
                imu_to_camera((raw.x, raw.y, raw.z), self._imu_rotation))
            if self._gravity is None:
                self._gravity = vector
            else:
                a: float = IMU_SMOOTHING
                self._gravity = tuple(                          # type: ignore[assignment]
                    previous + a * (current - previous)
                    for previous, current in zip(self._gravity, vector)
                )
        if self._gravity is None:
            return
        tilt, roll = orientation_from_gravity(*self._gravity)
        # Tilt gets no offset: there is no way to measure true tilt on site, so there would be
        # nothing to calibrate it against.
        self.settings.readings.tilt_measured = tilt
        self.settings.readings.roll_measured = roll - self.settings.readings.roll_offset

    def _setup_queues(self) -> None:
        self._setup_imu_queue()
        if self.do_stereo:
            if self.do_color:
                self.inputs[Input.COLOR_CONTROL] =  self.device.getInputQueue('color_control')
            self.inputs[Input.MONO_CONTROL] =       self.device.getInputQueue('mono_control')
            self.inputs[Input.STEREO_CONTROL] =     self.device.getInputQueue('stereo_control')
            self.outputs[Output.SYNC_FRAMES_OUT] =  self.device.getOutputQueue(name='sync', maxSize=1, blocking=False) # type: ignore
            self.outputs[Output.SYNC_FRAMES_OUT].addCallback(self._sync_callback)
            self.fps_counters[FrameType.VIDEO] = FPS(120)
            self.fps_counters[FrameType.LEFT_] = FPS(120)
            self.fps_counters[FrameType.RIGHT] = FPS(120)
            if self.show_stereo:
                self.fps_counters[FrameType.DEPTH] = FPS(120)
        elif self.do_color:
            self.inputs[Input.COLOR_CONTROL] =      self.device.getInputQueue('color_control')
            self.outputs[Output.VIDEO_FRAME_OUT] =  self.device.getOutputQueue(name='video', maxSize=1, blocking=False) # type: ignore
            self.outputs[Output.VIDEO_FRAME_OUT].addCallback(self._video_callback)
            self.fps_counters[FrameType.VIDEO] = FPS(120)
        else: # only mono
            self.inputs[Input.MONO_CONTROL] =       self.device.getInputQueue('mono_control')
            self.outputs[Output.VIDEO_FRAME_OUT] =  self.device.getOutputQueue(name='video', maxSize=1, blocking=False) # type: ignore
            self.outputs[Output.VIDEO_FRAME_OUT].addCallback(self._video_callback)
            self.fps_counters[FrameType.VIDEO] = FPS(120)
        if self.do_yolo:
            self.outputs[Output.TRACKLETS_OUT] = self.device.getOutputQueue(name='tracklets', maxSize=1, blocking=False) # type: ignore
            self.outputs[Output.TRACKLETS_OUT].addCallback(self._tracker_callback)

    def _close(self) -> None:
        self.settings.disconnect()
        self.device.close()
        for value in self.outputs.values():
            value.close()
        for value in self.inputs.values():
            value.close()

        self.frame_callbacks.clear()
        self.preview_callbacks.clear()
        self.sync_callbacks.clear()
        self.tracker_callbacks.clear()

        logger.info(f'{self.device_id} CLOSED')

    def _video_callback(self, msg: dai.ImgFrame) -> None:
        # print('RV', msg.getTimestamp())
        self._update_fps(FrameType.VIDEO)
        if self.do_color:
            self.settings.update_color_readback(msg)
        if self.do_stereo or not self.do_color:
            self.settings.update_mono_readback(msg)

        frame: ndarray = msg.getCvFrame()
        self._update_frame_callbacks(FrameType.VIDEO, frame)

    def _left_callback(self, msg: dai.ImgFrame) -> None:
        self._update_fps(FrameType.LEFT_)
        frame: ndarray = msg.getCvFrame()
        self._update_frame_callbacks(FrameType.LEFT_, frame)

    def _right_callback(self, msg: dai.ImgFrame) -> None:
        self._update_fps(FrameType.RIGHT)
        frame: ndarray = msg.getCvFrame()
        self._update_frame_callbacks(FrameType.RIGHT, frame)

    def _stereo_callback(self, msg: dai.ImgFrame) -> None:
        self._update_fps(FrameType.DEPTH)
        frame: ndarray = msg.getCvFrame()
        frame = applyColorMap(frame, COLORMAP_JET)
        self._update_frame_callbacks(FrameType.DEPTH, frame)

    def _sync_callback(self, message_group: dai.MessageGroup) -> None:
        frames = dict[FrameType, ndarray]()
        for name, msg in message_group:
            if type(msg) == dai.ImgFrame:
                if name == 'video':
                    # print(name, msg.getTimestampDevice(), message_group.getTimestampDevice(), msg.getSequenceNum(), self.cntr)
                    self._video_callback(msg)
                    frames[FrameType.VIDEO] = msg.getCvFrame()
                elif name == 'left':
                    name = 'left_'
                    # print(name, msg.getTimestampDevice(), message_group.getTimestampDevice(), msg.getSequenceNum(), self.cntr)
                    self._left_callback(msg)
                    frames[FrameType.LEFT_] = msg.getCvFrame()
                elif name == 'right':
                    # print(name, msg.getTimestampDevice(), message_group.getTimestampDevice(), msg.getSequenceNum(), self.cntr)
                    self._right_callback(msg)
                    frames[FrameType.RIGHT] = msg.getCvFrame()
                elif name == 'stereo':
                    self._stereo_callback(msg)
                else:
                    logger.info('unknown message', name)
        self._update_sync_callbacks(frames, self.fps)

        self.cntr = self.cntr + 1

    def _tracker_callback(self, msg: dai.RawTracklets) -> None:
        # print('RT', msg.getTimestamp()) # type: ignore
        self._update_tps()
        Ts: list[Tracklet] = msg.tracklets
        self.num_tracklets = len(Ts)
        self.settings.readings.tracklets = self.num_tracklets
        self._update_tracker_callbacks(Ts)

    # FPS
    def _update_fps(self, fps_type: FrameType) -> None:
        self.fps_counters[fps_type].processed()
        if fps_type == FrameType.VIDEO:
            self.settings.readings.video_fps = self.fps_counters[fps_type].get_rate_average()

    def _update_tps(self) -> None:
        self.tps_counter.processed()
        self.settings.readings.tracker_fps = self.tps_counter.get_rate_average()

    # CALLBACKS
    def _update_frame_callbacks(self, frame_type: FrameType, frame: ndarray) -> None:
        # if not self.running:
        #     return
        for c in self.frame_callbacks:
            c(self.id, frame_type, frame)
        if self.preview_type == frame_type:
            for c in self.preview_callbacks:
                c(self.id, frame_type, frame)
        if not self.do_stereo and frame_type == FrameType.VIDEO:
            frames: dict[FrameType, ndarray] = {}
            frames[frame_type] = frame
            self._update_sync_callbacks(frames, self.fps)

    def _update_sync_callbacks(self, frames: dict[FrameType, ndarray], fps: float) -> None:
        # if not self.running:
        #     return
        for c in self.sync_callbacks:
            c(self.id, frames, fps)

    def _update_tracker_callbacks(self, tracklets: list[Tracklet]) -> None:
        if not self.running:
            return
        for c in self.tracker_callbacks:
            c(self.id, tracklets)

    def add_frame_callback(self, callback: FrameCallback) -> None:
        if self.running:
            logger.warning('cannot add callback while camera is running')
            return
        if callback not in self.frame_callbacks:
            self.frame_callbacks.append(callback)

    def add_sync_callback(self, callback: SyncCallback) -> None:
        if self.running:
            logger.warning('cannot add callback while camera is running')
            return
        if callback not in self.sync_callbacks:
            self.sync_callbacks.append(callback)

    def add_preview_callback(self, callback: FrameCallback) -> None:
        if self.running:
            logger.warning('cannot add callback while camera is running')
            return
        if callback not in self.preview_callbacks:
            self.preview_callbacks.append(callback)

    def add_tracker_callback(self, callback: TrackerCallback) -> None:
        if self.running:
            logger.warning('cannot add callback while camera is running')
            return
        if callback not in self.tracker_callbacks:
            self.tracker_callbacks.append(callback)

    @staticmethod
    def _try_device(device_id: str, pipeline: dai.Pipeline, num_tries: int) -> dai.Device:
        device_info = dai.DeviceInfo(device_id)
        for attempt in range(num_tries):
            try:
                device = dai.Device(pipeline, device_info)
                return device
            except Exception as e:
                logger.warning("Attempt %s/%s - could not open camera: %s", attempt + 1, num_tries, e)
                continue
        raise Exception('Failed to open device after multiple attempts')
