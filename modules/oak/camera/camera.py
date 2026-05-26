# DOCS
# https://oak-web.readthedocs.io/
# https://docs.luxonis.com/software/depthai/examples/depth_post_processing/

import depthai as dai
from numpy import ndarray
from typing import Set
from threading import Thread, Event, Lock

from .pipeline import build_pipeline, get_model_path, PipelineConfig, PipelineHandles, PerspectiveConfig
from .definitions import *
from .settings import CameraSettings
from modules.utils import FPS

import logging
logger = logging.getLogger(__name__)

class Camera(Thread):
    _id_counter = 0
    _pipeline: dai.Pipeline | None = None
    _pipeline_handles: PipelineHandles | None = None
    _pipeline_lock: Lock = Lock()
    _MAX_CONSECUTIVE_FAILURES: int = 3

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
        self.do_yolo: bool =            core_settings.yolo
        self.do_720p: bool =            core_settings.hd_ready
        self.simulation: bool =         core_settings.sim_enabled

        self.perspective: PerspectiveConfig = PerspectiveConfig(
            core_settings.flip_h,
            core_settings.flip_v,
            core_settings.perspective
        )

        # DAI
        self.device:                    dai.Device
        self.inputs: dict[Input, dai.DataInputQueue] = {}
        self.outputs: dict[Output, dai.DataOutputQueue] = {}
        self.num_tracklets: int =       0
        self._pipeline_handles: PipelineHandles | None = None

        # FPS
        self.fps_counters: dict[FrameType, FPS] = {}
        self.tps_counter =              FPS(120)

        # FRAME TYPES
        self.frame_types: list[FrameType] = [FrameType.NONE_, FrameType.VIDEO]

        # CALLBACKS
        self.preview_callbacks: Set[FrameCallback] = set()
        self.frame_callbacks: Set[FrameCallback] = set()
        self.sync_callbacks: Set[SyncCallback] = set()
        self.tracker_callbacks: Set[TrackerCallback] = set()

        # PREVIEW
        self.preview_type =             FrameType.VIDEO

    def stop(self) -> None:
        self.running = False
        self.stop_event.set()

    def run(self) -> None:
        consecutive_failures: int = 0
        while not self.stop_event.is_set():
            try:
                if not self._open():
                    return
                consecutive_failures = 0
                self.stop_event.wait()
                self._close()
            except Exception:
                logger.exception("Camera error")
                consecutive_failures += 1
                if consecutive_failures >= Camera._MAX_CONSECUTIVE_FAILURES:
                    logger.error(f'Camera {self.device_id}: {consecutive_failures} consecutive failures, stopping')
                    return
                self.stop_event.wait(timeout=1.0)

    def _open(self) -> bool:
        device_list: list[str] = get_device_list(verbose=False)

        if self.device_id not in device_list:
            logger.warning(f'Camera: {self.device_id} NOT AVAILABLE in {device_list}')
            return False

        with Camera._pipeline_lock:
            if Camera._pipeline is None:
                Camera._pipeline = dai.Pipeline()
                self._setup_pipeline(Camera._pipeline)
                Camera._pipeline_handles = self._pipeline_handles
            else:
                self._pipeline_handles = Camera._pipeline_handles

        try:
            self.device = self._try_device(self.device_id, Camera._pipeline, num_tries=1)
        except Exception as e:
            logger.error(f'Could not open device: {e}')
            return False

        self._setup_queues()

        logger.info(f'Camera: {self.device_id} OPEN')
        self.running = True
        self.settings.connect(self.device, self.inputs, self.do_color)
        return True

    def _setup_pipeline(self, pipeline: dai.Pipeline) -> None:
        config = PipelineConfig(
            fps=self.fps,
            square=self.square,
            do_color=self.do_color,
            do_yolo=self.do_yolo,
            do_720p=self.do_720p,
            perspective=self.perspective,
            simulate=False,
            nn_path=get_model_path(self.model_path, self.square, False) if self.do_yolo else None,
        )
        self._pipeline_handles = build_pipeline(pipeline, config)

    def _setup_queues(self) -> None:
        assert self._pipeline_handles is not None
        handles = self._pipeline_handles
        if handles.do_color:
            self.inputs[Input.COLOR_CONTROL] =     self.device.getInputQueue('color_control')
        else:
            self.inputs[Input.MONO_CONTROL] =      self.device.getInputQueue('mono_control')
        video_q = self.device.getOutputQueue('video')
        video_q.setMaxSize(1)
        video_q.setBlocking(False)
        self.outputs[Output.VIDEO_FRAME_OUT] = video_q
        self.outputs[Output.VIDEO_FRAME_OUT].addCallback(self._video_callback)
        self.fps_counters[FrameType.VIDEO] = FPS(120)
        if handles.do_yolo:
            tracklets_q = self.device.getOutputQueue('tracklets')
            tracklets_q.setMaxSize(1)
            tracklets_q.setBlocking(False)
            self.outputs[Output.TRACKLETS_OUT] = tracklets_q
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

        logger.info(f'Camera: {self.device_id} CLOSED')

    def _video_callback(self, msg: dai.ImgFrame) -> None:
        # print('RV', msg.getTimestamp())
        self._update_fps(FrameType.VIDEO)
        if self.do_color:
            self.settings.update_color_readback(msg)
        if not self.do_color:
            self.settings.update_mono_readback(msg)

        frame: ndarray = msg.getCvFrame()
        self._update_frame_callbacks(FrameType.VIDEO, frame)

    def _tracker_callback(self, msg: dai.RawTracklets) -> None:
        # print('RT', msg.getTimestamp()) # type: ignore
        self._update_tps()
        Ts: list[Tracklet] = msg.tracklets
        self.num_tracklets = len(Ts)
        self.settings.tracklets = self.num_tracklets
        self._update_tracker_callbacks(Ts)

    # FPS
    def _update_fps(self, fps_type: FrameType) -> None:
        self.fps_counters[fps_type].processed()
        if fps_type == FrameType.VIDEO:
            self.settings.video_fps = self.fps_counters[fps_type].get_rate_average()

    def _update_tps(self) -> None:
        self.tps_counter.processed()
        self.settings.tracker_fps = self.tps_counter.get_rate_average()

    # CALLBACKS
    def _update_frame_callbacks(self, frame_type: FrameType, frame: ndarray) -> None:
        # if not self.running:
        #     return
        for c in self.frame_callbacks:
            c(self.id, frame_type, frame)
        if self.preview_type == frame_type:
            for c in self.preview_callbacks:
                c(self.id, frame_type, frame)
        if frame_type == FrameType.VIDEO:
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
            logger.warning('Camera: cannot add callback while camera is running')
            return
        self.frame_callbacks.add(callback)

    def add_sync_callback(self, callback: SyncCallback) -> None:
        if self.running:
            logger.warning('Camera: cannot add callback while camera is running')
            return
        self.sync_callbacks.add(callback)

    def add_preview_callback(self, callback: FrameCallback) -> None:
        if self.running:
            logger.warning('Camera: cannot add callback while camera is running')
            return
        self.preview_callbacks.add(callback)

    def add_tracker_callback(self, callback: TrackerCallback) -> None:
        if self.running:
            logger.warning('Camera: cannot add callback while camera is running')
            return
        self.tracker_callbacks.add(callback)

    @staticmethod
    def _try_device(device_id: str, pipeline: dai.Pipeline, num_tries: int) -> dai.Device:
        device_info = dai.DeviceInfo(device_id)
        for attempt in range(num_tries):
            try:
                device = dai.Device(pipeline, device_info)
                return device
            except Exception as e:
                print (f'Attempt {attempt + 1}/{num_tries} - could not open camera: {e}')
                continue
        raise Exception('Failed to open device after multiple attempts')











