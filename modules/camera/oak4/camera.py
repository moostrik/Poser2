"""The oak4 backend: one OAK4 camera on depthai 3, implementing `modules.camera.CameraSource`.

Each camera opens its own device and builds its own pipeline on it (`build_pipeline`; a depthai 3
pipeline is bound to one device), reads the delivered frames, the tracklets and the IMU through
queues on the node outputs, on depthai's callback thread, which is where the frames and tracklets
become the contract's and get their host time. The device runs the chain `model_chain` names.
DETECTOR gives boxes only; with a pose chain a tracklet carries its keypoints: from its own
detection in the one-model chain, or, in the two-model chain, from the pose of its crop, joined to
it by frame (`PoseJoin`). With `depth_model` it also delivers depth aligned to the frames, as DEPTH
frames. The sensor controls are `SensorControl`'s for as long as the device is open.

The frame leaves here untouched where it can: an NV12 frame goes on packed, and the GPU upload
converts, straightens and mirrors it, so this thread does no color work. The camera cannot do
either to that frame itself, so its boxes and keypoints get the same here to match. The IMU's tilt
makes the homography every frame carries (`TiltCorrection`), on this one view for color and depth
alike. `readings.video_latency`,
`tracker_latency` and `depth_latency` say what each message took from capture to here.

Docs: https://docs.luxonis.com/software-v3/depthai
"""

import logging
import math
import time
from pathlib import Path
from threading import Thread, Event
from typing import cast

import depthai as dai
from numpy import ndarray

from .. import CameraResolution, CameraFrame, FrameReadings, RigCameraReadings, CameraTracklet, FrameType, CameraFrameCallback, \
    TrackletCallback, orientation_from_gravity, frame_window, frame_coverage, coverage_summary, \
    delivered_height, lens_field, lens_deviation, imu_to_camera, unroll_imu_frame, TiltCorrection, corrected_roi, corrected_keypoints
from modules.utils import FPS, Rect

from .control import DepthControl, SensorControl
from .definitions import SENSOR_SOCKET, requested_size, get_device_list, find_device, link_resolution, log_connected_sensors, \
    read_lens_calibration, imu_rotation_to_camera, IMU_SMOOTHING, IMU_BOARD_ROLL, ModelChain, CameraLink, DepthModel, \
    model_archives, TRACKING_STATUS, tracklet_keypoints, keypoints_in_frame, packed_nv12, smoothed_latency, \
    mirrored_roi, mirrored_keypoints
from .pipeline import build_pipeline, delivered_size, MountConfig, PipelineOutputs
from .pose_join import PoseJoin, Box
from .settings import Oak4CameraSettings

logger = logging.getLogger(__name__)

QUEUE_SIZE: int = 1         # the newest frame only: a frame the host did not take in time is dropped, not queued
RETRY_INTERVAL: float = 1.0     # s between two tries to find and open the camera
FRAME_TIMEOUT: float = 5.0      # s without a frame while open: the camera is opened again
FIRST_FRAME_GRACE: float = 10.0 # s on top of the timeout for the first frame, while the models load
FRAME_CHECK: float = 0.5        # s between two looks at the time of the last frame


class Camera(Thread):
    _id_counter = 0

    def __init__(self, settings: Oak4CameraSettings) -> None:
        super().__init__()
        self.stop_event = Event()
        self.running: bool = False
        self._problem: str = ''             # why there are no frames, as logged; '' while they come
        self._last_frame: float = 0.0

        # ID
        self.id: int = Camera._id_counter
        Camera._id_counter += 1

        # SETTINGS (reactive)
        self.settings: Oak4CameraSettings = settings

        # FIXED SETTINGS (read from INIT fields once)
        self.device_id: str =           settings.device_id
        self.link: CameraLink =         settings.link
        self.model_path: str =          settings.model_path
        self.fps: float =               settings.fps
        self.color: bool =              settings.color
        self.depth_model: DepthModel =  settings.depth_model

        self.mount: MountConfig = MountConfig(
            flip_h=settings.flip_h,
            flip_v=settings.flip_v,
            fov_h=settings.fov,
            lens_fov=settings.lens_fov,
            lens_center_x=settings.lens_center_x,
            lens_center_y=settings.lens_center_y,
        )
        self.resolution: CameraResolution = settings.resolution     # until `_take_link` knows the link
        self.frame_height: int = 0
        self.frame_size: tuple[int, int] = (0, 0)
        self._set_resolution(settings.resolution)
        self.model_chain: ModelChain = settings.model_chain
        models = Path(self.model_path)
        pose, detector = model_archives(self.model_chain)
        self.pose_archive: Path | None = models / pose if pose else None
        self.detector_archive: Path | None = models / detector if detector else None
        self.pose_join: PoseJoin | None = PoseJoin() if pose and detector else None

        # DEVICE — its pipeline and queues are made in `_open` and closed with it.
        self.device: dai.Device | None = None
        self.pipeline: dai.Pipeline | None = None
        self.queues: list[dai.MessageQueue] = []
        self.control: SensorControl | None = None
        self.depth_control: DepthControl | None = None
        self.num_tracklets: int = 0
        self.flip_on_host: bool = False
        self._frame_transformation: dai.ImgTransformation | None = None     # the delivered frame's, for the crops' keypoints

        # FPS
        self.video_fps: FPS = FPS(120)
        self.tracklet_fps: FPS = FPS(120)
        self.depth_fps: FPS = FPS(120)

        # CALLBACKS (run in registration order)
        self.frame_callbacks: list[CameraFrameCallback] = []
        self.tracklet_callbacks: list[TrackletCallback] = []

        # MOUNT READOUT — the gravity vector, smoothed, and how to get it into the camera's frame.
        self._imu_rotation: list[list[float]] | None = None
        self._gravity: tuple[float, float, float] | None = None

        # TILT CORRECTION — the direct frame's lens, read off its first frame.
        self._intrinsics: list[list[float]] | None = None
        self.tilt_correction: TiltCorrection = TiltCorrection(settings.tilt)

    @property
    def readings(self) -> RigCameraReadings:
        return self.settings.readings

    @property
    def connected(self) -> bool:
        return self.device is not None

    def stop(self) -> None:
        self.running = False
        self.stop_event.set()

    def run(self) -> None:
        # The thread opens and starts over when the camera was not there or its frames stopped: one
        # just closed by the last run comes free by itself, a lost one when it is back
        while not self.stop_event.is_set():
            try:
                if self._open():
                    self._watch_frames()
                    self._close()
                    logger.info(f'{self.device_id} CLOSED')
            except Exception as e:
                self._warn(f'camera error: {e}')
            self.stop_event.wait(RETRY_INTERVAL)
        self.frame_callbacks.clear()
        self.tracklet_callbacks.clear()

    def _watch_frames(self) -> None:
        """Until told to stop, or until the frames have stopped: a cable, its power, a crashed device."""
        while not self.stop_event.wait(FRAME_CHECK):
            if time.perf_counter() - self._last_frame > FRAME_TIMEOUT:
                self._warn(f'no frame for {FRAME_TIMEOUT:.0f} s, opening the camera again')
                return

    def _warn(self, problem: str) -> None:
        """A reason there are no frames: logged when it is a new one, not at every retry."""
        if problem != self._problem:
            self._problem = problem
            logger.warning(problem)

    # ── Open and close ─────────────────────────────────────────────────

    def _open(self) -> bool:
        device_list: list[str] = get_device_list(verbose=False)
        if not self.device_id and device_list:
            self.device_id = device_list[0]          # no id in the preset: the first OAK4 found
            logger.info(f'no device id set: opening the first OAK4 found, {self.device_id} (of {device_list})')
        if not self.device_id:
            self._warn('no OAK4 found (an OAK-D is never opened by this backend)')
            return False
        if self.device_id not in device_list:
            self._warn(f'{self.device_id} NOT AVAILABLE in the OAK4s found: {device_list}')
            return False
        info: dai.DeviceInfo | None = find_device(self.device_id, self.link)
        if info is None:
            self._warn(f'{self.device_id} is gone')
            return False

        self._take_link(info)
        self._log_frame_geometry()
        try:
            self.device = dai.Device(info)
        except Exception as e:
            self._warn(f'{self.device_id} could not open: {e}')
            return False
        try:
            self.pipeline = dai.Pipeline(self.device)
            outputs: PipelineOutputs = build_pipeline(self.pipeline, self.fps, self.color, self.resolution,
                                                      self.mount, frame_height=self.frame_height,
                                                      pose_archive=self.pose_archive,
                                                      detector_archive=self.detector_archive,
                                                      depth_model=self.depth_model)
            self._setup_queues(outputs)
            self.pipeline.start()
        except Exception as e:                       # the camera is there and refuses the setup: a retry changes nothing
            logger.error(f'{self.device_id} pipeline failed, not trying again: {e}')
            self._close()
            self.stop_event.set()
            return False

        logger.info(f'{self.device_id} OPEN over {info.protocol}')
        log_connected_sensors(self.device, self.device_id)
        self._read_mount_calibration()
        self._last_frame = time.perf_counter() + FIRST_FRAME_GRACE
        self.running = True
        return True

    def _set_resolution(self, resolution: CameraResolution) -> None:
        self.resolution = resolution
        self.frame_height = delivered_height(self.color, resolution, self.mount.fov_h, 0.0, self.mount.lens_fov, self.mount.lens_center)
        self.frame_size = delivered_size(self.color, resolution, self.frame_height)

    def _take_link(self, info: dai.DeviceInfo) -> None:
        """The frame this link carries. A reopen can land on the other link, so the lens read off the
        last frame and the homography made from it go too: the next frame brings them back."""
        self._set_resolution(link_resolution(info.protocol, self.settings.resolution, self.settings.ethernet_resolution))
        self._intrinsics = None
        self.tilt_correction.update(math.nan, None, self.frame_size)

    def _close(self) -> None:
        self.running = False
        if self.control is not None:
            self.control.close()
            self.control = None
        if self.depth_control is not None:
            self.depth_control.close()
            self.depth_control = None
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
                self.pipeline.wait()
            except Exception as exc:
                logger.debug(f'{self.device_id} pipeline stop: {exc}')
            self.pipeline = None
        for queue in self.queues:
            queue.close()
        self.queues.clear()
        if self.device is not None:
            try:
                self.device.close()
            except Exception as exc:
                logger.debug(f'{self.device_id} device close: {exc}')
            self.device = None

        readings: RigCameraReadings = self.settings.readings        # they update on a frame, and none come now
        readings.video_fps = readings.video_min_fps = readings.video_latency = 0.0
        readings.tracker_fps = readings.tracker_latency = readings.depth_fps = readings.depth_latency = 0.0
        readings.cpu = readings.memory = readings.temperature = 0.0
        readings.tracklets = 0

    def _setup_queues(self, outputs: PipelineOutputs) -> None:
        """The host's ends of the pipeline: the frames every camera has, the tracklets with a model,
        the depth with a depth model, the IMU, the camera's load, the sensor controls."""
        self.flip_on_host = outputs.flip_on_host
        video: dai.MessageQueue = outputs.video.createOutputQueue(maxSize=QUEUE_SIZE, blocking=False)
        video.addCallback(self._video_callback)
        self.queues.append(video)
        if outputs.tracklets is not None:
            tracklets: dai.MessageQueue = outputs.tracklets.createOutputQueue(maxSize=QUEUE_SIZE, blocking=False)
            tracklets.addCallback(self._tracker_callback)
            self.queues.append(tracklets)
        if outputs.poses is not None:
            poses: dai.MessageQueue = outputs.poses.createOutputQueue(maxSize=QUEUE_SIZE, blocking=False)
            poses.addCallback(self._poses_callback)
            self.queues.append(poses)
        if outputs.depth is not None:
            depth: dai.MessageQueue = outputs.depth.createOutputQueue(maxSize=QUEUE_SIZE, blocking=False)
            depth.addCallback(self._depth_callback)
            self.queues.append(depth)
        imu: dai.MessageQueue = outputs.imu.createOutputQueue(maxSize=QUEUE_SIZE, blocking=False)
        imu.addCallback(self._imu_callback)
        self.queues.append(imu)
        system: dai.MessageQueue = outputs.system.createOutputQueue(maxSize=QUEUE_SIZE, blocking=False)
        system.addCallback(self._system_callback)
        self.queues.append(system)
        control: dai.InputQueue = outputs.control.createInputQueue()
        assert self.device is not None
        self.control = SensorControl(self.settings, self.device, control, self.flip_on_host)
        if outputs.depth_config is not None:
            self.depth_control = DepthControl(self.settings.depth, outputs.depth_config.createInputQueue())

    def _log_frame_geometry(self) -> None:
        """The window the warp delivers and where the sensor covers it, pure geometry from the
        preset, logged so a rig check reads the horizon row off the log."""
        src: tuple[int, int] = requested_size(self.color, self.resolution)
        out: tuple[int, int] = self.frame_size
        window = frame_window(src, out, src[0], self.mount.fov_h, 0.0,
                              self.mount.lens_fov, self.mount.lens_center)
        coverage = frame_coverage(src, out, src[0], self.mount.fov_h, 0.0,
                                  self.mount.flip_h, self.mount.flip_v,
                                  self.mount.lens_fov, self.mount.lens_center)
        logger.info(f'frame {out[0]}x{out[1]}: elevation {window.elevation_bottom:+.1f} to '
                    f'{window.elevation_top:+.1f} deg, horizon at row {window.horizon_px:.1f}, '
                    f'{window.focal:.1f} px/rad; '
                    f'{coverage_summary(coverage, out, src[0], self.mount.fov_h, self.mount.flip_h)}')

    def _read_mount_calibration(self) -> None:
        """One-shot at open: what this unit says about its own lens and its IMU's orientation."""
        assert self.device is not None
        src: tuple[int, int] = requested_size(self.color, self.resolution)
        lens = read_lens_calibration(self.device, SENSOR_SOCKET, src, self.device_id)
        if lens is not None:
            focal, cx, cy = lens
            field: float = lens_field(focal, src[0])
            error: float = lens_deviation(focal, cx, cy, src, src[0], self.mount.fov_h,
                                          self.mount.lens_fov, self.mount.lens_center)
            self.settings.readings.fov_factory = field
            self.settings.readings.lens_error = error
            logger.info(f'{self.device_id} lens: field {field:.1f} deg across {src[0]} px, '
                        f'centre offset ({cx - (src[0] - 1) / 2.0:+.1f}, {cy - (src[1] - 1) / 2.0:+.1f}) px; '
                        f'{error:.2f} deg off the shared lens')
        self._imu_rotation = imu_rotation_to_camera(self.device, SENSOR_SOCKET, self.device_id)

    # ── The queues' callbacks: the hand-over to the contract ───────────

    def _video_callback(self, name: str, msg: dai.ImgFrame) -> None:
        self._last_frame = time.perf_counter()
        if self._problem:
            logger.info(f'{self.device_id} delivers frames again')
            self._problem = ''
        readings: RigCameraReadings = self.settings.readings
        readings.video_latency = smoothed_latency(readings.video_latency, msg)
        self.video_fps.processed()
        readings.video_fps = self.video_fps.get_rate_average()
        readings.video_min_fps = self.video_fps.get_rate_minimum()
        if self.control is not None:
            self.control.readback(msg)
        if self.pose_join is not None:
            self._frame_transformation = msg.getTransformation()
        if msg.getType() == dai.ImgFrame.Type.NV12:
            if self._intrinsics is None:
                self._intrinsics = msg.getTransformation().getIntrinsicMatrix()
                logger.info(f'{self.device_id} delivered lens: {self._intrinsics}')
            self._emit_frame(FrameType.VIDEO, packed_nv12(msg), nv12=True, readings=self._frame_readings())
        else:
            self._emit_frame(FrameType.VIDEO, msg.getCvFrame(), readings=self._frame_readings())

    def _frame_readings(self) -> FrameReadings:
        readings: RigCameraReadings = self.settings.readings
        return FrameReadings(readings.tilt_measured, readings.roll_measured, self.settings.sensor.exposure, self.settings.sensor.iso)

    def _depth_callback(self, name: str, msg: dai.ImgFrame) -> None:
        """The hand-over of the aligned depth: uint16 millimetres, on the frames' own view."""
        readings: RigCameraReadings = self.settings.readings
        readings.depth_latency = smoothed_latency(readings.depth_latency, msg)
        self.depth_fps.processed()
        readings.depth_fps = self.depth_fps.get_rate_average()
        self._emit_frame(FrameType.DEPTH, msg.getFrame().copy())

    def _tracklet(self, tracklet: dai.Tracklet, keypoints: ndarray | None, timestamp: float) -> CameraTracklet:
        """One device tracklet as the contract's, straightened and mirrored when the frames are."""
        roi = Rect(tracklet.roi.x, tracklet.roi.y, tracklet.roi.width, tracklet.roi.height)
        homography: ndarray | None = self.tilt_correction.homography
        if homography is not None:
            roi, keypoints = corrected_roi(roi, homography), corrected_keypoints(keypoints, homography)
        if self.flip_on_host:
            roi, keypoints = mirrored_roi(roi), mirrored_keypoints(keypoints)
        return CameraTracklet(cam_id=self.id, id=tracklet.id, status=TRACKING_STATUS[tracklet.status], roi=roi,
                              age=tracklet.age, timestamp=timestamp, keypoints=keypoints)

    def _tracker_callback(self, name: str, msg: dai.Tracklets) -> None:
        """The hand-over of the device's tracklets: the contract's, with the keypoints the pose model
        found for each, and the message's host time. In the two-model chain the keypoints come with
        the poses (`_poses_callback`), and the tracklets wait in the join for them."""
        self.settings.readings.tracker_latency = smoothed_latency(self.settings.readings.tracker_latency, msg)
        self.tracklet_fps.processed()
        self.settings.readings.tracker_fps = self.tracklet_fps.get_rate_average()
        now: float = time.perf_counter()
        one_model: bool = self.pose_join is None
        tracklets: list[CameraTracklet] = [
            self._tracklet(t, tracklet_keypoints(t.srcImgDetection) if one_model else None, now)
            for t in msg.tracklets]
        self.num_tracklets = len(tracklets)
        self.settings.readings.tracklets = self.num_tracklets
        if self.pose_join is None:
            self._emit_tracklets(tracklets, now)
            return
        boxes: list[Box] = [(t.srcImgDetection.xmin, t.srcImgDetection.ymin, t.srcImgDetection.xmax,
                             t.srcImgDetection.ymax) for t in msg.tracklets]
        joined = self.pose_join.put_tracklets(msg.getSequenceNum(), tracklets, boxes)
        if joined is not None:
            self._emit_tracklets(joined, now)

    def _poses_callback(self, name: str, msg) -> None:
        """The two-model chain's poses: depthai-nodes' `GatheredData`, the people's detections of one
        frame (`reference_data`) with one keypoint message per person (`items`, in the same order),
        each remapped from its crop into the delivered frame."""
        if self.pose_join is None:
            return
        detections = msg.reference_data
        boxes: list[Box] = [(d.xmin, d.ymin, d.xmax, d.ymax) for d in detections.detections]
        keypoints = [keypoints_in_frame(item, self._frame_transformation) for item in msg.items]
        if len(keypoints) != len(boxes):
            logger.debug(f'{self.device_id} frame {detections.getSequenceNum()}: {len(keypoints)} poses for {len(boxes)} people')
            keypoints = (keypoints + [None] * len(boxes))[:len(boxes)]
        homography: ndarray | None = self.tilt_correction.homography
        if homography is not None:
            keypoints = [corrected_keypoints(points, homography) for points in keypoints]
        if self.flip_on_host:
            keypoints = [mirrored_keypoints(points) for points in keypoints]
        joined = self.pose_join.put_poses(detections.getSequenceNum(), boxes, keypoints)
        if joined is not None:
            self._emit_tracklets(joined, time.perf_counter())

    def _system_callback(self, name: str, msg: dai.SystemInformationRVC4) -> None:
        readings: RigCameraReadings = self.settings.readings
        readings.cpu = msg.cpuAvgUsage.average * 100.0
        readings.memory = msg.ddrMemoryUsage.used / msg.ddrMemoryUsage.total * 100.0
        readings.temperature = msg.chipTemperature.average

    def _imu_callback(self, name: str, msg: dai.IMUData) -> None:
        for packet in msg.packets:
            raw = packet.acceleroMeter
            vector: tuple[float, float, float] = unroll_imu_frame(
                imu_to_camera((raw.x, raw.y, raw.z), self._imu_rotation), IMU_BOARD_ROLL)
            if self._gravity is None:
                self._gravity = vector
            else:
                a: float = IMU_SMOOTHING
                self._gravity = cast(tuple[float, float, float], tuple(
                    previous + a * (current - previous)
                    for previous, current in zip(self._gravity, vector)
                ))
        if self._gravity is None:
            return
        tilt, roll = orientation_from_gravity(*self._gravity)
        self.settings.readings.tilt_measured = tilt
        self.settings.readings.roll_measured = roll - self.settings.readings.roll_offset
        self.tilt_correction.update(tilt, self._intrinsics, self.frame_size)

    def _emit_frame(self, kind: FrameType, image: ndarray, nv12: bool = False, readings: FrameReadings | None = None) -> None:
        """The hand-over: the frame gets its host time here and goes to every callback."""
        frame = CameraFrame(cam_id=self.id, kind=kind, image=image, timestamp=time.perf_counter(),
                            nv12=nv12, flip_h=self.flip_on_host, homography=self.tilt_correction.homography,
                            readings=readings, intrinsics=self._intrinsics)
        for c in self.frame_callbacks:
            c(frame)

    def _emit_tracklets(self, tracklets: list[CameraTracklet], timestamp: float) -> None:
        if not self.running:
            return
        for c in self.tracklet_callbacks:
            c(self.id, tracklets, timestamp)

    # ── CameraSource ───────────────────────────────────────────────────

    def add_frame_callback(self, callback: CameraFrameCallback) -> None:
        if self.running:
            logger.warning('cannot add callback while camera is running')
            return
        if callback not in self.frame_callbacks:
            self.frame_callbacks.append(callback)

    def add_tracklet_callback(self, callback: TrackletCallback) -> None:
        if self.running:
            logger.warning('cannot add callback while camera is running')
            return
        if callback not in self.tracklet_callbacks:
            self.tracklet_callbacks.append(callback)
