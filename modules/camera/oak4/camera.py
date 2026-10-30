"""The oak4 backend: one OAK4 camera on depthai 3, implementing `modules.camera.CameraSource`.

Each camera opens its own device and builds its own pipeline on it (`build_pipeline`; a depthai 3
pipeline is bound to one device), reads the delivered frames, the tracklets and the IMU through
queues on the node outputs, on depthai's callback thread, which is where the frames and tracklets
become the contract's and get their host time. With `yolo` the device runs the pose model, so a
tracklet carries its keypoints: from its own detection in the one-model chain, or, in the two-model
chain, from the pose of its crop, joined to it by frame (`PoseJoin`). The sensor controls are
`SensorControl`'s for as long as the device is open.

Docs: https://docs.luxonis.com/software-v3/depthai
"""

import logging
import time
from pathlib import Path
from threading import Thread, Event
from typing import cast

import depthai as dai
from numpy import ndarray

from .. import CameraResolution, CameraFrame, RigCameraReadings, CameraTracklet, FrameType, CameraFrameCallback, \
    TrackletCallback, orientation_from_gravity, frame_size, frame_window, frame_coverage, coverage_summary, \
    delivered_height, lens_field, lens_deviation, imu_to_camera, unroll_imu_frame
from modules.utils import FPS, Rect

from .control import SensorControl
from .definitions import SENSOR_SOCKET, requested_size, get_device_list, log_connected_sensors, \
    read_lens_calibration, imu_rotation_to_camera, IMU_SMOOTHING, IMU_BOARD_ROLL, POSE_ARCHIVE, DETECTOR_ARCHIVE, \
    TOP_DOWN_ARCHIVE, PoseChain, TRACKING_STATUS, tracklet_keypoints, keypoints_in_frame
from .pipeline import build_pipeline, MountConfig, PipelineOutputs
from .pose_join import PoseJoin, Box
from .settings import Oak4CameraSettings

logger = logging.getLogger(__name__)

QUEUE_SIZE: int = 1         # the newest frame only: a frame the host did not take in time is dropped, not queued


class Camera(Thread):
    _id_counter = 0

    def __init__(self, settings: Oak4CameraSettings) -> None:
        super().__init__()
        self.stop_event = Event()
        self.running: bool = False

        # ID
        self.id: int = Camera._id_counter
        Camera._id_counter += 1

        # SETTINGS (reactive)
        self.settings: Oak4CameraSettings = settings

        # FIXED SETTINGS (read from INIT fields once)
        self.device_id: str =           settings.device_id
        self.model_path: str =          settings.model_path
        self.fps: float =               settings.fps
        self.square: bool =             settings.square
        self.color: bool =              settings.color
        self.yolo: bool =               settings.yolo
        self.resolution: CameraResolution = settings.resolution

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
        self.frame_height: int = delivered_height(self.color, self.resolution, self.mount.fov_h, self.mount.tilt,
                                                  self.mount.lens_fov, self.mount.lens_center)
        self.pose_chain: PoseChain = settings.pose_chain
        models = Path(self.model_path)
        two_models: bool = self.yolo and self.pose_chain == PoseChain.TWO_MODELS
        self.pose_archive: Path | None = models / (TOP_DOWN_ARCHIVE if two_models else POSE_ARCHIVE) if self.yolo else None
        self.detector_archive: Path | None = models / DETECTOR_ARCHIVE if two_models else None
        self.pose_join: PoseJoin | None = PoseJoin() if two_models else None

        # DEVICE — its pipeline and queues are made in `_open` and closed with it.
        self.device: dai.Device | None = None
        self.pipeline: dai.Pipeline | None = None
        self.queues: list[dai.MessageQueue] = []
        self.control: SensorControl | None = None
        self.num_tracklets: int = 0
        self._frame_transformation: dai.ImgTransformation | None = None     # the delivered frame's, for the crops' keypoints

        # FPS
        self.video_fps: FPS = FPS(120)
        self.tracklet_fps: FPS = FPS(120)

        # CALLBACKS (run in registration order)
        self.frame_callbacks: list[CameraFrameCallback] = []
        self.tracklet_callbacks: list[TrackletCallback] = []

        # MOUNT READOUT — the gravity vector, smoothed, and how to get it into the camera's frame.
        self._imu_rotation: list[list[float]] | None = None
        self._gravity: tuple[float, float, float] | None = None

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
        if not self.device_id and device_list:
            self.device_id = device_list[0]          # no id in the preset: the first OAK4 on the network
            logger.info(f'no device id set: opening the first OAK4 found, {self.device_id} (of {device_list})')
        if not self.device_id:
            logger.warning('no OAK4 found (an OAK-D is never opened by this backend)')
            return False
        if self.device_id not in device_list:
            logger.warning(f'{self.device_id} NOT AVAILABLE in the OAK4s found: {device_list}')
            return False

        self._log_frame_geometry()
        try:
            self.device = dai.Device(dai.DeviceInfo(self.device_id))
            self.pipeline = dai.Pipeline(self.device)
            outputs: PipelineOutputs = build_pipeline(self.pipeline, self.fps, self.color, self.square, self.resolution,
                                                      self.mount, frame_height=self.frame_height,
                                                      pose_archive=self.pose_archive,
                                                      detector_archive=self.detector_archive)
            self._setup_queues(outputs)
            self.pipeline.start()
        except Exception as e:
            logger.error(f'{self.device_id} could not open: {e}')
            self._close()
            return False

        logger.info(f'{self.device_id} OPEN')
        log_connected_sensors(self.device, self.device_id)
        self._read_mount_calibration()
        self.running = True
        return True

    def _close(self) -> None:
        if self.control is not None:
            self.control.close()
            self.control = None
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
            self.device.close()
            self.device = None

        self.frame_callbacks.clear()
        self.tracklet_callbacks.clear()

        logger.info(f'{self.device_id} CLOSED')

    def _setup_queues(self, outputs: PipelineOutputs) -> None:
        """The host's ends of the pipeline: the frames every camera has, the tracklets with a pose
        model, the IMU, the sensor controls."""
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
        imu: dai.MessageQueue = outputs.imu.createOutputQueue(maxSize=QUEUE_SIZE, blocking=False)
        imu.addCallback(self._imu_callback)
        self.queues.append(imu)
        control: dai.InputQueue = outputs.control.createInputQueue()
        self.control = SensorControl(self.settings, control)

    def _log_frame_geometry(self) -> None:
        """The window the warp delivers and where the sensor covers it, pure geometry from the
        preset, logged so a rig check reads the horizon row off the log. Not on the keystone path."""
        if self.mount.keystone != 0.0 and self.mount.tilt == 0.0:
            return
        src: tuple[int, int] = requested_size(self.color, self.resolution)
        out: tuple[int, int] = frame_size(self.color, self.resolution, self.square, self.frame_height)
        window = frame_window(src, out, src[0], self.mount.fov_h, self.mount.tilt,
                              self.mount.lens_fov, self.mount.lens_center)
        coverage = frame_coverage(src, out, src[0], self.mount.fov_h, self.mount.tilt,
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
        self.video_fps.processed()
        self.settings.readings.video_fps = self.video_fps.get_rate_average()
        self.settings.readings.video_min_fps = self.video_fps.get_rate_minimum()
        if self.control is not None:
            self.control.readback(msg)
        if self.pose_join is not None:
            self._frame_transformation = msg.getTransformation()
        self._emit_frame(FrameType.VIDEO, msg.getCvFrame())

    def _tracker_callback(self, name: str, msg: dai.Tracklets) -> None:
        """The hand-over of the device's tracklets: the contract's, with the keypoints the pose model
        found for each, and the message's host time. In the two-model chain the keypoints come with
        the poses (`_poses_callback`), and the tracklets wait in the join for them."""
        self.tracklet_fps.processed()
        self.settings.readings.tracker_fps = self.tracklet_fps.get_rate_average()
        now: float = time.perf_counter()
        one_model: bool = self.pose_join is None
        tracklets: list[CameraTracklet] = [
            CameraTracklet(cam_id=self.id, id=t.id, status=TRACKING_STATUS[t.status],
                           roi=Rect(t.roi.x, t.roi.y, t.roi.width, t.roi.height), age=t.age, timestamp=now,
                           keypoints=tracklet_keypoints(t.srcImgDetection) if one_model else None)
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
        joined = self.pose_join.put_poses(detections.getSequenceNum(), boxes, keypoints)
        if joined is not None:
            self._emit_tracklets(joined, time.perf_counter())

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

    def _emit_frame(self, kind: FrameType, image: ndarray) -> None:
        """The hand-over: the frame gets its host time here and goes to every callback."""
        frame = CameraFrame(cam_id=self.id, kind=kind, image=image, timestamp=time.perf_counter())
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
