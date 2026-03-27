# DOCS
# https://oak-web.readthedocs.io/
# https://docs.luxonis.com/software/depthai/examples/depth_post_processing/

from __future__ import annotations
import depthai as dai
from cv2 import applyColorMap, COLORMAP_JET
from numpy import ndarray
from typing import Set
from threading import Thread, Event

from modules.cam.depthcam.Pipeline import setup_pipeline, get_frame_types, PerspectiveConfig, get_stereo_config
from modules.cam.depthcam.Definitions import *
from modules.cam.depthcam.CoreSettings import CoreSettings
from modules.utils.FPS import FPS

class Core(Thread):
    _id_counter = 0
    _pipeline: dai.Pipeline | None = None

    def __init__(self, core_settings: CoreSettings) -> None:
        super().__init__()
        self.stop_event = Event()
        self.running: bool = False

        # ID
        self.id: int =                  Core._id_counter
        Core._id_counter +=             1
        self.id_string: str =           str(self.id)

        # SETTINGS (reactive)
        self.settings: CoreSettings =   core_settings

        # FIXED SETTINGS (read from INIT fields once)
        self.device_id: str =           core_settings.device_id
        self.model_path: str =          core_settings.model_path
        self.fps: float =               core_settings.fps
        self.square: bool =             core_settings.square
        self.do_color: bool =           core_settings.color
        self.do_stereo: bool =          core_settings.stereo
        self.do_yolo: bool =            core_settings.yolo
        self.do_720p: bool =            core_settings.hd_ready
        self.show_stereo: bool =        core_settings.show_stereo
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

        # FPS
        self.fps_counters: dict[FrameType, FPS] = {}
        self.tps_counter =              FPS(120)

        # FRAME TYPES
        self.frame_types: list[FrameType] = get_frame_types(self.do_color, self.do_stereo, self.show_stereo, core_settings.sim_enabled)
        self.frame_types.sort(key=lambda x: x.value)

        # CALLBACKS
        self.preview_callbacks: Set[FrameCallback] = set()
        self.frame_callbacks: Set[FrameCallback] = set()
        self.sync_callbacks: Set[SyncCallback] = set()
        self.tracker_callbacks: Set[TrackerCallback] = set()

        # PREVIEW
        self.preview_type =             FrameType.VIDEO

        # STEREO CONFIG
        self.stereo_config: dai.RawStereoDepthConfig = get_stereo_config(self.do_color)

        self.cntr: int = 0

    def stop(self) -> None:
        self.running = False
        self.stop_event.set()

    def run(self) -> None:
        while not self.stop_event.is_set():
            if not self._open():
                return
            self.stop_event.wait()
            self._close()

    def _open(self) -> bool:
        device_list: list[str] = get_device_list(verbose=False)

        if self.device_id not in device_list:
            print(f'Camera: {self.device_id} NOT AVAILABLE in {device_list}')
            return False

        if Core._pipeline is None:
            Core._pipeline = dai.Pipeline()
            self._setup_pipeline(Core._pipeline)

        try:
            self.device = self._try_device(self.device_id, Core._pipeline, num_tries=1)
        except Exception as e:
            print(f'Could not open device: {e}')
            return False

        self._setup_queues()

        print(f'Camera: {self.device_id} OPEN')
        self.running = True
        self._bind_settings()
        self._apply_settings()
        return True

    def _setup_pipeline(self, pipeline: dai.Pipeline) -> None:
            setup_pipeline(pipeline, self.model_path, self.fps, self.square, self.do_color, self.do_stereo, self.do_yolo, self.do_720p, self.show_stereo, self.perspective, simulate=False)

    def _setup_queues(self) -> None:
        if self.do_stereo:
            if self.do_color:
                self.inputs[Input.COLOR_CONTROL] =  self.device.getInputQueue('color_control')
            self.inputs[Input.MONO_CONTROL] =       self.device.getInputQueue('mono_control')
            self.inputs[Input.STEREO_CONTROL] =     self.device.getInputQueue('stereo_control')
            self.outputs[Output.SYNC_FRAMES_OUT] =  self.device.getOutputQueue(name='sync', maxSize=1, blocking=False)
            self.outputs[Output.SYNC_FRAMES_OUT].addCallback(self._sync_callback)
            self.fps_counters[FrameType.VIDEO] = FPS(120)
            self.fps_counters[FrameType.LEFT_] = FPS(120)
            self.fps_counters[FrameType.RIGHT] = FPS(120)
            if self.show_stereo:
                self.fps_counters[FrameType.DEPTH] = FPS(120)
        elif self.do_color:
            self.inputs[Input.COLOR_CONTROL] =      self.device.getInputQueue('color_control')
            self.outputs[Output.VIDEO_FRAME_OUT] =  self.device.getOutputQueue(name='video', maxSize=1, blocking=False)
            self.outputs[Output.VIDEO_FRAME_OUT].addCallback(self._video_callback)
            self.fps_counters[FrameType.VIDEO] = FPS(120)
        else: # only mono
            self.inputs[Input.MONO_CONTROL] =       self.device.getInputQueue('mono_control')
            self.outputs[Output.VIDEO_FRAME_OUT] =  self.device.getOutputQueue(name='video', maxSize=1, blocking=False)
            self.outputs[Output.VIDEO_FRAME_OUT].addCallback(self._video_callback)
            self.fps_counters[FrameType.VIDEO] = FPS(120)
        if self.do_yolo:
            self.outputs[Output.TRACKLETS_OUT] = self.device.getOutputQueue(name='tracklets', maxSize=1, blocking=False)
            self.outputs[Output.TRACKLETS_OUT].addCallback(self._tracker_callback)

    def _close(self) -> None:
        self.device.close()
        for value in self.outputs.values():
            value.close()
        for value in self.inputs.values():
            value.close()

        self.frame_callbacks.clear()
        self.preview_callbacks.clear()
        self.sync_callbacks.clear()
        self.tracker_callbacks.clear()

        print(f'Camera: {self.device_id} CLOSED')

    # ── Settings binding ──────────────────────────────────────────────

    def _bind_settings(self) -> None:
        """Bind reactive WRITE fields → DAI hardware commands."""
        s = self.settings

        # Color controls
        s.bind(CoreSettings.color_auto_exposure, self._on_color_auto_exposure)
        s.bind(CoreSettings.color_exposure, self._on_color_exposure)
        s.bind(CoreSettings.color_iso, self._on_color_iso)
        s.bind(CoreSettings.color_auto_balance, self._on_color_auto_balance)
        s.bind(CoreSettings.color_balance, self._on_color_balance)
        s.bind(CoreSettings.color_brightness, self._on_color_brightness)
        s.bind(CoreSettings.color_contrast, self._on_color_contrast)
        s.bind(CoreSettings.color_saturation, self._on_color_saturation)
        s.bind(CoreSettings.color_luma_denoise, self._on_color_luma_denoise)
        s.bind(CoreSettings.color_sharpness, self._on_color_sharpness)

        # Mono controls
        s.bind(CoreSettings.mono_auto_exposure, self._on_mono_auto_exposure)
        s.bind(CoreSettings.mono_exposure, self._on_mono_exposure)
        s.bind(CoreSettings.mono_iso, self._on_mono_iso)

        # IR controls
        s.bind(CoreSettings.ir_flood_light, self._on_ir_flood_light)
        s.bind(CoreSettings.ir_grid_light, self._on_ir_grid_light)

        # Stereo controls
        s.bind(CoreSettings.stereo_depth_min, self._on_stereo_config)
        s.bind(CoreSettings.stereo_depth_max, self._on_stereo_config)
        s.bind(CoreSettings.stereo_brightness_min, self._on_stereo_config)
        s.bind(CoreSettings.stereo_brightness_max, self._on_stereo_config)
        s.bind(CoreSettings.stereo_median_filter, self._on_stereo_config)

    def _apply_settings(self) -> None:
        """Push all current field values to hardware (called once after open)."""
        self._on_color_auto_exposure(self.settings.color_auto_exposure)
        if not self.settings.color_auto_exposure:
            self._send_color_exposure_iso(self.settings.color_exposure, self.settings.color_iso)
        self._on_color_auto_balance(self.settings.color_auto_balance)
        if not self.settings.color_auto_balance:
            self._on_color_balance(self.settings.color_balance)
        self._on_color_brightness(self.settings.color_brightness)
        self._on_color_contrast(self.settings.color_contrast)
        self._on_color_saturation(self.settings.color_saturation)
        self._on_color_luma_denoise(self.settings.color_luma_denoise)
        self._on_color_sharpness(self.settings.color_sharpness)
        self._on_mono_auto_exposure(self.settings.mono_auto_exposure)
        if not self.settings.mono_auto_exposure:
            self._send_mono_exposure_iso(self.settings.mono_exposure, self.settings.mono_iso)
        self._on_stereo_config()
        if not self.do_color:
            self._on_ir_flood_light(self.settings.ir_flood_light)
            self._on_ir_grid_light(self.settings.ir_grid_light)

    # ── Color callbacks ───────────────────────────────────────────────

    def _on_color_auto_exposure(self, value=None) -> None:
        if not self.running: return
        if self.settings.color_auto_exposure:
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureEnable()
            self._send_control(Input.COLOR_CONTROL, ctrl)
        else:
            self._send_color_exposure_iso(self.settings.color_exposure, self.settings.color_iso)

    def _send_color_exposure_iso(self, exposure: int, iso: int) -> None:
        if not self.running: return
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(exposure, iso)
        self._send_control(Input.COLOR_CONTROL, ctrl)

    def _on_color_exposure(self, value: int = 0) -> None:
        if not self.settings.color_auto_exposure:
            self._send_color_exposure_iso(self.settings.color_exposure, self.settings.color_iso)

    def _on_color_iso(self, value: int = 0) -> None:
        if not self.settings.color_auto_exposure:
            self._send_color_exposure_iso(self.settings.color_exposure, self.settings.color_iso)

    def _on_color_auto_balance(self, value=None) -> None:
        if not self.running: return
        if self.settings.color_auto_balance:
            ctrl = dai.CameraControl()
            ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
            self._send_control(Input.COLOR_CONTROL, ctrl)
        else:
            self._on_color_balance(self.settings.color_balance)

    def _on_color_balance(self, value: int = 0) -> None:
        if not self.running: return
        ctrl = dai.CameraControl()
        ctrl.setManualWhiteBalance(self.settings.color_balance)
        self._send_control(Input.COLOR_CONTROL, ctrl)

    def _on_color_brightness(self, value: int = 0) -> None:
        if not self.running: return
        ctrl = dai.CameraControl()
        ctrl.setBrightness(self.settings.color_brightness)
        self._send_control(Input.COLOR_CONTROL, ctrl)

    def _on_color_contrast(self, value: int = 0) -> None:
        if not self.running: return
        ctrl = dai.CameraControl()
        ctrl.setContrast(self.settings.color_contrast)
        self._send_control(Input.COLOR_CONTROL, ctrl)

    def _on_color_saturation(self, value: int = 0) -> None:
        if not self.running: return
        ctrl = dai.CameraControl()
        ctrl.setSaturation(self.settings.color_saturation)
        self._send_control(Input.COLOR_CONTROL, ctrl)

    def _on_color_luma_denoise(self, value: int = 0) -> None:
        if not self.running: return
        ctrl = dai.CameraControl()
        ctrl.setLumaDenoise(self.settings.color_luma_denoise)
        self._send_control(Input.COLOR_CONTROL, ctrl)

    def _on_color_sharpness(self, value: int = 0) -> None:
        if not self.running: return
        ctrl = dai.CameraControl()
        ctrl.setSharpness(self.settings.color_sharpness)
        self._send_control(Input.COLOR_CONTROL, ctrl)

    # ── Mono callbacks ────────────────────────────────────────────────

    def _on_mono_auto_exposure(self, value=None) -> None:
        if not self.running: return
        if self.settings.mono_auto_exposure:
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureEnable()
            self._send_control(Input.MONO_CONTROL, ctrl)
        else:
            self._send_mono_exposure_iso(self.settings.mono_exposure, self.settings.mono_iso)

    def _send_mono_exposure_iso(self, exposure: int, iso: int) -> None:
        if not self.running: return
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(exposure, iso)
        self._send_control(Input.MONO_CONTROL, ctrl)

    def _on_mono_exposure(self, value: int = 0) -> None:
        if not self.settings.mono_auto_exposure:
            self._send_mono_exposure_iso(self.settings.mono_exposure, self.settings.mono_iso)

    def _on_mono_iso(self, value: int = 0) -> None:
        if not self.settings.mono_auto_exposure:
            self._send_mono_exposure_iso(self.settings.mono_exposure, self.settings.mono_iso)

    # ── IR callbacks ──────────────────────────────────────────────────

    def _on_ir_flood_light(self, value: float = 0.0) -> None:
        if not self.running or self.do_color: return
        self.device.setIrFloodLightIntensity(self.settings.ir_flood_light)

    def _on_ir_grid_light(self, value: float = 0.0) -> None:
        if not self.running or self.do_color: return
        self.device.setIrLaserDotProjectorIntensity(self.settings.ir_grid_light)

    # ── Stereo callback ───────────────────────────────────────────────

    def _on_stereo_config(self, value=None) -> None:
        if not self.running: return
        s = self.settings
        self.stereo_config.postProcessing.thresholdFilter.minRange = s.stereo_depth_min
        self.stereo_config.postProcessing.thresholdFilter.maxRange = s.stereo_depth_max
        self.stereo_config.postProcessing.brightnessFilter.minBrightness = s.stereo_brightness_min
        self.stereo_config.postProcessing.brightnessFilter.maxBrightness = s.stereo_brightness_max
        mf = s.stereo_median_filter
        if mf == StereoMedianFilterType.OFF:
            self.stereo_config.postProcessing.median = dai.MedianFilter.MEDIAN_OFF
        elif mf == StereoMedianFilterType.KERNEL_3x3:
            self.stereo_config.postProcessing.median = dai.MedianFilter.KERNEL_3x3
        elif mf == StereoMedianFilterType.KERNEL_5x5:
            self.stereo_config.postProcessing.median = dai.MedianFilter.KERNEL_5x5
        elif mf == StereoMedianFilterType.KERNEL_7x7:
            self.stereo_config.postProcessing.median = dai.MedianFilter.KERNEL_7x7
        self._send_control(Input.STEREO_CONTROL, self.stereo_config)

    # ── Readback (READ fields written from camera frames) ─────────────

    def _update_color_readback(self, frame: dai.ImgFrame) -> None:
        if self.settings.color_auto_exposure:
            self.settings.actual_color_exposure = int(frame.getExposureTime().total_seconds() * 1000000)
            self.settings.actual_color_iso = frame.getSensitivity()
        if self.settings.color_auto_balance:
            self.settings.actual_color_balance = frame.getColorTemperature()

    def _update_mono_readback(self, frame: dai.ImgFrame) -> None:
        if self.settings.mono_auto_exposure:
            self.settings.actual_mono_exposure = int(frame.getExposureTime().total_seconds() * 1000000)
            self.settings.actual_mono_iso = frame.getSensitivity()

    def _video_callback(self, msg: dai.ImgFrame) -> None:
        # print('RV', msg.getTimestamp())
        self._update_fps(FrameType.VIDEO)
        if self.do_color:
            self._update_color_readback(msg)
        if self.do_stereo or not self.do_color:
            self._update_mono_readback(msg)

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
                    print('unknown message', name)
        self._update_sync_callbacks(frames, self.fps)

        self.cntr = self.cntr + 1

    def _tracker_callback(self, msg: dai.RawTracklets) -> None:
        # print('RT', msg.getTimestamp()) # type: ignore
        self._update_tps()
        Ts: list[Tracklet] = msg.tracklets
        self.num_tracklets = len(Ts)
        self.settings.num_tracklets = self.num_tracklets
        self._update_tracker_callbacks(Ts)

    # FPS
    def _update_fps(self, fps_type: FrameType) -> None:
        self.fps_counters[fps_type].processed()
        if fps_type == FrameType.VIDEO:
            self.settings.fps_video = self.fps_counters[fps_type].get_rate_average()

    def _update_tps(self) -> None:
        self.tps_counter.processed()
        self.settings.tps = self.tps_counter.get_rate_average()

    # CONTROL
    def _send_control(self, input: Input, control) -> None:
        if not self.running:
            return
        if input in self.inputs:
            self.inputs[input].send(control)

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
            print('Camera: cannot add callback while camera is running')
            return
        self.frame_callbacks.add(callback)

    def add_sync_callback(self, callback: SyncCallback) -> None:
        if self.running:
            print('Camera: cannot add callback while camera is running')
            return
        self.sync_callbacks.add(callback)

    def add_preview_callback(self, callback: FrameCallback) -> None:
        if self.running:
            print('Camera: cannot add callback while camera is running')
            return
        self.preview_callbacks.add(callback)

    def add_tracker_callback(self, callback: TrackerCallback) -> None:
        if self.running:
            print('Camera: cannot add callback while camera is running')
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











