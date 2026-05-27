# DOCS
# https://oak-web.readthedocs.io/
# https://docs.luxonis.com/software/depthai/examples/depth_post_processing/

import time
import depthai as dai
from numpy import ndarray
from typing import Set
from threading import Thread, Event, Lock, Barrier, BrokenBarrierError

from ._pipeline import build_pipeline, get_model_path, PipelineConfig, PipelineHandles, PerspectiveConfig
from ._definitions import *
from .settings import CameraSettings
from modules.utils import FPS

import logging
logger = logging.getLogger(__name__)

class UsbCamera(Thread):
    _id_counter: int = 0

    # Phase B lock: serialises pipeline.start() calls across all UsbCamera threads so the
    # USB host isn't hit by concurrent XLink stream setup. Shared across all instances.
    _start_lock: Lock = Lock()

    def __init__(self, core_settings: CameraSettings, barrier: Barrier | None = None, device_info: DeviceInfo | None = None) -> None:
        super().__init__()
        self.stop_event = Event()
        self.running: bool = False

        # ID
        self.id: int =                  UsbCamera._id_counter
        UsbCamera._id_counter +=        1
        self.id_string: str =           str(self.id)

        self._barrier = barrier
        self._device_info = device_info

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
        self._device: dai.Device | None =       None
        self._pipeline: dai.Pipeline | None =   None
        self.inputs: dict =                     {}
        self.outputs: dict =                    {}
        self.num_tracklets: int =               0
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
        try:
            if self._open():
                self._poll_loop()
        except Exception:
            logger.exception(f'Camera {self.device_id}: error')
        finally:
            self._close()

    def _poll_loop(self) -> None:
        """depthai v3 host-side XLink reader requires the queue to be actively
        drained by the host. `addCallback` does not keep the stream alive
        reliably under v3 — the probe uses blocking `q.get()` polling, which
        works. Each UsbCamera owns its own thread (this one) and drives its
        queues here."""
        video_q = self.outputs.get(Output.VIDEO_FRAME_OUT)
        tracklets_q = self.outputs.get(Output.TRACKLETS_OUT)
        while not self.stop_event.is_set():
            if video_q is not None:
                msg = video_q.tryGet()
                if msg is not None:
                    self._video_callback(msg)
            if tracklets_q is not None:
                tmsg = tracklets_q.tryGet()
                if tmsg is not None:
                    self._tracker_callback(tmsg)
            # Tiny sleep to avoid spinning at 100% CPU when no frame is ready.
            # 1 ms is short enough not to bottleneck a 30 fps stream.
            time.sleep(0.001)

    def _open(self) -> bool:
        info = self._device_info
        if info is None:
            logger.warning(f'Camera: {self.device_id} NOT AVAILABLE in {get_device_list()}')
            if self._barrier is not None:
                try:
                    self._barrier.wait(timeout=60.0)
                except BrokenBarrierError:
                    pass
            return False

        # Phase A — concurrent, single attempt, no sleep.
        # Boot + build + queues for all cameras happen simultaneously so USB
        # enumerations overlap. No camera is streaming while a peer enumerates.
        # NO retries here: sleeping between attempts while a peer has a booted-but-idle
        # device is the exact root cause of the XLink error. If boot fails, signal
        # ready and return False; run() will retry after the barrier is already
        # satisfied (safe — no idle peers at that point).
        try:
            device = dai.Device(info)
        except RuntimeError as e:
            logger.warning(f'Camera {self.device_id}: boot failed: {e}')
            if self._barrier is not None:
                try:
                    self._barrier.wait(timeout=60.0)
                except BrokenBarrierError:
                    pass
            return False
        self._device = device
        try:
            self._pipeline = dai.Pipeline(device)
            self._setup_pipeline(self._pipeline)
            self._setup_queues()
        except Exception:
            logger.exception(f'Camera {self.device_id}: pipeline build failed')
            if self._barrier is not None:
                try:
                    self._barrier.wait(timeout=60.0)
                except BrokenBarrierError:
                    pass
            return False

        # Barrier — wait for every UsbCamera to finish Phase A before any start().
        if self._barrier is not None:
            try:
                self._barrier.wait(timeout=60.0)
            except BrokenBarrierError:
                logger.error(f'Camera {self.device_id}: timed out waiting for peers')
                return False

        # Settle delay — let USB bus quiesce after all cameras finish Phase A.
        # Prevents "Device already closed" errors on pipeline.start() caused by
        # the XLink connection dropping between boot and start.
        time.sleep(0.5)

        # Phase B — sequential pipeline.start() under _start_lock.
        with UsbCamera._start_lock:
            try:
                self._pipeline.start()
            except Exception:
                logger.exception(f'Camera {self.device_id}: pipeline.start() failed')
                return False

        logger.info(f'Camera: {self.device_id} OPEN')
        self.running = True
        self.settings.connect(self._device, self.inputs, self.do_color)
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
        if handles.control_in is not None:
            if handles.do_color:
                self.inputs[Input.COLOR_CONTROL] = handles.control_in.createInputQueue()
            else:
                self.inputs[Input.MONO_CONTROL] = handles.control_in.createInputQueue()
        video_q = handles.video_out.createOutputQueue(maxSize=4, blocking=False)
        self.outputs[Output.VIDEO_FRAME_OUT] = video_q
        self.fps_counters[FrameType.VIDEO] = FPS(120)
        if handles.do_yolo and handles.tracklets_out is not None:
            tracklets_q = handles.tracklets_out.createOutputQueue(maxSize=4, blocking=False)
            self.outputs[Output.TRACKLETS_OUT] = tracklets_q

    def _close(self) -> None:
        self.running = False
        try:
            self.settings.disconnect()
        except Exception:
            logger.exception(f'Camera {self.device_id}: settings.disconnect failed')
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                logger.exception(f'Camera {self.device_id}: pipeline.stop failed')
            self._pipeline = None
        self._device = None
        self.inputs.clear()
        self.outputs.clear()
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

    def _tracker_callback(self, msg: dai.Tracklets) -> None:
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
