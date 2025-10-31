# DOCS
# https://oak-web.readthedocs.io/
# https://docs.luxonis.com/software/depthai/examples/depth_post_processing/

from __future__ import annotations
import depthai as dai
from cv2 import applyColorMap, COLORMAP_JET
from numpy import ndarray
from typing import Set
from threading import Thread, Event

from modules.Settings import Settings
from modules.cam.depthcam.Pipeline import setup_pipeline, get_frame_types, PerspectiveConfig
from modules.cam.depthcam.Definitions import *
from modules.cam.depthcam.CoreSettings import CoreSettings
from modules.cam.depthcam.Gui import Gui
from modules.utils.FPS import FPS

class Core(Thread):
    _id_counter = 0
    _pipeline: dai.Pipeline | None = None

    def __init__(self, gui, device_id: str, general_settings:Settings) -> None:
        super().__init__()
        self.stop_event = Event()
        self.running: bool = False

        # ID
        self.id: int =                  Core._id_counter
        Core._id_counter +=             1
        self.id_string: str =           str(self.id)
        self.device_id: str =           device_id

        # FIXED SETTINGS
        self.model_path: str =          general_settings.path_model
        self.fps: float =               general_settings.camera_fps
        self.square: bool =             general_settings.camera_square
        self.do_color: bool =           general_settings.camera_color
        self.do_stereo: bool =          general_settings.camera_stereo
        self.do_yolo: bool =            general_settings.camera_yolo
        self.show_stereo: bool =        general_settings.camera_show_stereo
        self.simulation: bool =         general_settings.camera_simulation

        self.perspective: PerspectiveConfig = PerspectiveConfig(
            general_settings.camera_flip_h,
            general_settings.camera_flip_v,
            general_settings.camera_perspective
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
        self.frame_types: list[FrameType] = get_frame_types(self.do_color, self.do_stereo, self.show_stereo, general_settings.camera_simulation)
        self.frame_types.sort(key=lambda x: x.value)

        # CALLBACKS
        self.preview_callbacks: Set[FrameCallback] = set()
        self.frame_callbacks: Set[FrameCallback] = set()
        self.sync_callbacks: Set[SyncCallback] = set()
        self.tracker_callbacks: Set[TrackerCallback] = set()

        # SETTINGS
        self.preview_type =             FrameType.VIDEO
        self.settings: CoreSettings =   CoreSettings(self)
        self.gui: Gui =                 Gui(gui, self.settings, general_settings)

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
            print(f'Camera: {self.device_id} NOT AVAILABLE')
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
        self.settings.apply_settings()
        return True

    def _setup_pipeline(self, pipeline: dai.Pipeline) -> None:
            setup_pipeline(pipeline, self.model_path, self.fps, self.square, self.do_color, self.do_stereo, self.do_yolo, self.show_stereo, self.perspective, simulate=False)

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

    def _video_callback(self, msg: dai.ImgFrame) -> None:
        # print('RV', msg.getTimestamp())
        self._update_fps(FrameType.VIDEO)
        self.gui.update_from_frame()
        if self.do_color:
            self.settings.update_color_control(msg)
        if self.do_stereo or not self.do_color:
            self.settings.update_mono_control(msg)

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
        fps: float = self.fps_counters[FrameType.VIDEO].get_rate_average()
        self._update_sync_callbacks(frames, fps)

        self.cntr = self.cntr + 1

    def _tracker_callback(self, msg: dai.RawTracklets) -> None:
        # print('RT', msg.getTimestamp()) # type: ignore
        self._update_tps()
        Ts: list[Tracklet] = msg.tracklets
        self.num_tracklets = len(Ts)
        self.gui.update_from_tracker()
        self._update_tracker_callbacks(Ts)

    # FPS
    def _update_fps(self, fps_type: FrameType) -> None:
        self.fps_counters[fps_type].processed()

    def _update_tps(self) -> None:
        self.tps_counter.processed()

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
            self._update_sync_callbacks(frames, 30.0)

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











