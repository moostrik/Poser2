# DOCS
# https://oak-web.readthedocs.io/
# https://docs.luxonis.com/software/depthai/examples/depth_post_processing/

from __future__ import annotations
import depthai as dai
from cv2 import applyColorMap, COLORMAP_JET
from numpy import ndarray
from typing import Set
from threading import Thread, Event

from modules.cam.depthcam.Pipeline import setup_pipeline, get_frame_types
from modules.cam.depthcam.Definitions import *
from modules.cam.depthcam.Settings import Settings
from modules.cam.depthcam.Gui import Gui
from modules.utils.FPS import FPS

class Core(Thread):
    _id_counter = 0
    _pipeline: dai.Pipeline | None = None

    def __init__(self, gui, device_id: str, model_path:str, fps: int = 30,
                 do_color: bool = True, do_stereo: bool = True, do_person: bool = True,
                 lowres: bool = False, show_stereo: bool = False) -> None:

        super().__init__()
        self.stop_event = Event()

        # ID
        self.id: int =                  Core._id_counter
        Core._id_counter +=             1
        self.id_string: str =           str(self.id)
        self.device_id: str =           device_id

        # FIXED SETTINGS
        self.model_path: str =          model_path
        self.fps: int =                 fps
        self.do_color: bool =           do_color
        self.do_stereo: bool =          do_stereo
        self.do_person: bool =          do_person
        self.lowres: bool =             lowres
        self.show_stereo: bool =        show_stereo

        # DAI
        self.device_open: bool =        False
        self.device:                    dai.Device
        self.color_control:             dai.DataInputQueue
        self.mono_control:              dai.DataInputQueue
        self.stereo_control:            dai.DataInputQueue
        self.frame_queue:               dai.DataOutputQueue
        self.tracklet_queue:            dai.DataOutputQueue
        self.num_tracklets: int =       0

        # FPS
        self.fps_counter =              FPS(120)
        self.tps_counter =              FPS(120)

        # FRAME TYPES
        self.frame_types: list[FrameType] = get_frame_types(do_color, do_stereo, show_stereo)
        self.frame_types.sort(key=lambda x: x.value)

        # CALLBACKS
        self.preview_callbacks: Set[PreviewCallback] = set()
        self.tracker_callbacks: Set[TrackerCallback] = set()
        self.fps_callbacks: Set[FPSCallback] = set()
        self.frame_callbacks: dict[FrameType, Set[FrameCallback]] = {}
        for t in self.frame_types:
            if t == FrameType.NONE: continue
            self.frame_callbacks[t] = set()

        # SETTINGS
        self.preview_type =             FrameType.VIDEO
        self.settings: Settings =       Settings(self)
        self.gui: Gui =                 Gui(gui, self.settings)

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        while not self.stop_event.is_set():
            self._open()
            self.stop_event.wait()
            self._close()

    def _open(self) -> None:
        device_list: list[str] = Core.get_device_list(verbose=False)
        if self.device_id not in device_list:
            print(f'Camera: {self.device_id} NOT AVAILABLE')
            return

        if Core._pipeline is None:
            Core._pipeline = dai.Pipeline()
            setup_pipeline(Core._pipeline, self.model_path, self.fps, self.do_color, self.do_stereo, self.do_person, self.lowres, self.show_stereo)

        try:
            self.device = self._try_device(self.device_id, Core._pipeline, num_tries=3)
        except Exception as e:
            print(f'Could not open device: {e}')
            return

        self.frame_queue =      self.device.getOutputQueue(name='output_images', maxSize=4, blocking=False)
        self.tracklet_queue =   self.device.getOutputQueue(name='tracklets', maxSize=4, blocking=False)
        self.color_control =    self.device.getInputQueue('color_control')
        self.mono_control =     self.device.getInputQueue('mono_control')
        self.stereo_control =   self.device.getInputQueue('stereo_control')

        self.frame_queue.addCallback(self._frame_callback)
        self.tracklet_queue.addCallback(self._tracker_callback)

        print(f'Camera: {self.device_id} OPEN')
        self.device_open: bool =        True

    def _close(self) -> None:
        if not self.device_open: return
        self.device_open = False

        self.device.close()
        self.stereo_control.close()
        self.mono_control.close()
        self.color_control.close()
        self.frame_queue.close()
        self.tracklet_queue.close()

        self.frame_callbacks.clear()
        self.preview_callbacks.clear()
        self.tracker_callbacks.clear()

        print(f'Camera: {self.device_id} CLOSED')

    def _frame_callback(self, message_group: dai.MessageGroup) -> None:
        self._update_fps()

        for name, msg in message_group:
            if type(msg) == dai.ImgFrame:
                frame: ndarray = msg.getCvFrame()
                if name == 'video':
                    self.settings.update_color_control(msg)
                    self._update_callbacks(FrameType.VIDEO, frame)

                elif name == 'left':
                    self.settings.update_mono_control(msg)
                    self._update_callbacks(FrameType.LEFT, frame)

                elif name == 'right':
                    self._update_callbacks(FrameType.RIGHT, frame)

                elif name == 'stereo':
                    frame = applyColorMap(frame, COLORMAP_JET)
                    self._update_callbacks(FrameType.STEREO, frame)

                self.gui.update_from_frame()

            else:
                print('unknown message', name)

    def _tracker_callback(self, msg: dai.RawTracklets) -> None:
        self._update_tps()
        Ts: list[Tracklet] = msg.tracklets
        self.num_tracklets = len(Ts)
        self.gui.update_from_tracker()
        for t in Ts:
            for c in self.tracker_callbacks:
                c(self.id, t)

    # FPS
    def _update_fps(self) -> None:
        self.fps_counter.processed()
        for c in self.fps_callbacks:
            c(self.id, self.fps_counter.get_rate_average())

    def _update_tps(self) -> None:
        self.tps_counter.processed()

    # CALLBACKS
    def _update_callbacks(self, frame_type: FrameType, frame: ndarray) -> None:
        for c in self.frame_callbacks[frame_type]:
            c(self.id, frame_type, frame)
        if self.preview_type == frame_type:
            for c in self.preview_callbacks:
                c(self.id, frame)

    def add_frame_callback(self, frame_type: FrameType, callback: FrameCallback) -> None:
        self.frame_callbacks[frame_type].add(callback)
    def discard_frame_callback(self, frameType: FrameType, callback: FrameCallback) -> None:
        self.frame_callbacks[frameType].discard(callback)
    def clear_frame_callbacks(self) -> None:
        self.frame_callbacks.clear()

    def add_preview_callback(self, callback: PreviewCallback) -> None:
        self.preview_callbacks.add(callback)
    def discard_preview_callback(self, callback: PreviewCallback) -> None:
        self.preview_callbacks.discard(callback)
    def clear_preview_callbacks(self) -> None:
        self.preview_callbacks.clear()

    def add_tracker_callback(self, callback: TrackerCallback) -> None:
        self.tracker_callbacks.add(callback)
    def discard_tracker_callback(self, callback: TrackerCallback) -> None:
        self.tracker_callbacks.discard(callback)
    def clear_tracker_callbacks(self) -> None:
        self.tracker_callbacks.clear()

    def add_fps_callback(self, callback: FPSCallback) -> None:
        self.fps_callbacks.add(callback)
    def discard_fps_callback(self, callback: FPSCallback) -> None:
        self.fps_callbacks.discard(callback)
    def clear_fps_callbacks(self) -> None:
        self.fps_callbacks.clear()

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

    @staticmethod
    def get_device_list(verbose: bool = False) -> list[str]:
        device_list: list[str] = []
        if verbose:
            print('-- CAMERAS --------------------------------------------------')
        for device in dai.Device.getAllAvailableDevices():
            device_list.append(device.getMxId())
            if verbose:
                print(f"Camera: {device.getMxId()} {device.state}")
        if verbose:
            print('-------------------------------------------------------------')
        return device_list









