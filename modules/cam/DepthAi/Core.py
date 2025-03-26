# DOCS
# https://oak-web.readthedocs.io/
# https://docs.luxonis.com/software/depthai/examples/depth_post_processing/

import depthai as dai
from cv2 import applyColorMap, COLORMAP_JET
from numpy import ndarray
from typing import Set
from threading import Thread, Event

from modules.cam.DepthAi.Pipeline import setup_pipeline, get_frame_types
from modules.cam.DepthAi.Definitions import *
from modules.utils.FPS import FPS

class DepthAiCore(Thread):
    _id_counter = 0

    def __init__(self, model_path:str, fps: int = 30,
                 do_color: bool = True, do_stereo: bool = True, do_person: bool = True,
                 lowres: bool = False, show_stereo: bool = False) -> None:

        super().__init__()
        self.stop_event = Event()

        # ID
        self.id: int =                  DepthAiCore._id_counter
        self.id_string: str =           str(self.id)
        DepthAiCore._id_counter +=      1

        # FIXED SETTINGS
        self.model_path: str =          model_path
        self.fps: int =                 fps
        self.do_color: bool =           do_color
        self.do_stereo: bool =          do_stereo
        self.do_person: bool =          do_person
        self.lowres: bool =             lowres
        self.show_stereo: bool =        show_stereo

        # GENERAL SETTINGS
        self.preview_type =             FrameType.VIDEO

        # DAI
        self.device:                    dai.Device
        self.color_control:             dai.DataInputQueue
        self.mono_control:              dai.DataInputQueue
        self.stereo_control:            dai.DataInputQueue
        self.frame_queue:               dai.DataOutputQueue
        self.frame_callback_id:         int
        self.tracklet_queue:            dai.DataOutputQueue
        self.tracklet_callback_id:      int
        self.device_open: bool =        False
        self.num_tracklets: int =       0

        # FPS
        self.fps_counter =              FPS(120)
        self.tps_counter =              FPS(120)

        # FRAME TYPES
        frame_types: list[FrameType] = get_frame_types(do_color, do_stereo, show_stereo)
        frame_types.sort(key=lambda x: x.value)

        # CALLBACKS
        self.preview_callbacks: Set[PreviewCallback] = set()
        self.tracker_callbacks: Set[TrackerCallback] = set()
        self.fps_callbacks: Set[FPSCallback] = set()
        self.frame_callbacks: dict[FrameType, Set[FrameCallback]] = {}
        for t in frame_types:
            if t == FrameType.NONE: continue
            self.frame_callbacks[t] = set()

    def __exit__(self) -> None:
        self._close()

    def stop(self) -> None:
        self.stop_event.set()
        self.join()

    def run(self) -> None:
        while not self.stop_event.is_set():
            if not self.device_open:
                self._open()
            self.stop_event.wait()
            self._close()

    def _open(self) -> bool:
        if self.device_open: return True

        pipeline = dai.Pipeline()
        setup_pipeline(pipeline, self.model_path, self.fps, self.do_color, self.do_stereo, self.do_person, self.lowres, self.show_stereo)

        try: self.device = dai.Device(pipeline)
        except Exception as e:
            print('could not open camera:', e, '... Try again')
            try: self.device = dai.Device(pipeline)
            except Exception as e:
                print('still could not open camera:', e, '... Try again')
                try: self.device = dai.Device(pipeline)
                except Exception as e:
                    print('still could not open camera:', e, '... Giving up')
                    return False

        self.frame_queue =       self.device.getOutputQueue(name='output_images', maxSize=4, blocking=False)
        self.tracklet_queue =    self.device.getOutputQueue(name='tracklets', maxSize=4, blocking=False)
        self.color_control =     self.device.getInputQueue('color_control')
        self.mono_control =      self.device.getInputQueue('mono_control')
        self.stereo_control =    self.device.getInputQueue('stereo_control')

        self.frame_callback_id = self.frame_queue.addCallback(self._frame_callback)
        self.tracklet_callback_id = self.tracklet_queue.addCallback(self._tracker_callback)

        self.device_open = True
        return True

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


    def _frame_callback(self, message_group: dai.MessageGroup) -> None:
        self._update_fps()

        for name, msg in message_group:
            if name == 'video':
                self._update_color_control(msg)
                frame: ndarray = msg.getCvFrame() #type:ignore
                self._update_callbacks(FrameType.VIDEO, frame)

            elif name == 'left':
                self._update_mono_control(msg)
                frame: ndarray = msg.getCvFrame() #type:ignore
                self._update_callbacks(FrameType.LEFT, frame)

            elif name == 'right':
                frame = msg.getCvFrame() #type:ignore
                self._update_callbacks(FrameType.RIGHT, frame)

            elif name == 'stereo':
                frame = msg.getCvFrame() #type:ignore
                frame = applyColorMap(frame, COLORMAP_JET)
                self._update_callbacks(FrameType.STEREO, frame)

            else:
                print('unknown message', name)

    def _tracker_callback(self, msg) -> None:
        self._update_tps()
        Ts = msg.tracklets
        self.num_tracklets = len(Ts)
        for t in Ts:
            # tracklet: Tracklet = Tracklet.from_dai(t, self.ID)
            for c in self.tracker_callbacks:
                c(self.id, t)

    def _update_color_control(self, frame) -> None:
        pass

    def _update_mono_control(self, frame) -> None:
        pass

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








