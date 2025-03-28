from modules.cam.depthcam.Core import *
from modules.cam.depthcam.Definitions import FrameType
from modules.cam.depthcam.Pipeline import get_stereo_config, get_frame_types
from modules.cam.depthplayer.SyncPlayer import SyncPlayer
from modules.cam.depthplayer.Player import HwAccelerationType
from cv2 import resize
import time

class CorePlayer(Core):

    def __init__(self, gui, syncplayer: SyncPlayer, device_id: str, model_path:str, fps: int = 30,
                 do_color: bool = True, do_stereo: bool = True, do_person: bool = True,
                 lowres: bool = False, show_stereo: bool = False) -> None:
        super().__init__(gui, device_id, model_path, fps, do_color, do_stereo, do_person, lowres, show_stereo)

        self.sync_player: SyncPlayer = syncplayer
        self.ex_video:  dai.DataInputQueue
        self.ex_left:   dai.DataInputQueue
        self.ex_right:  dai.DataInputQueue

    def start(self) -> None: # override
        self.sync_player.addFrameCallback(self._video_frame_callback)
        super().start()

    def stop(self) -> None: # override
        self.sync_player.discardFrameCallback(self._video_frame_callback)
        super().stop()

    # TEMPORARY OVERRIDE TO TEST PLAYBACK
    # def run(self) -> None: # override
    #     while not self.stop_event.is_set():
    #         self.stop_event.wait()

    def _open(self) -> None:
        device_list: list[str] = Core.get_device_list(verbose=False)
        if self.device_id not in device_list:
            print(f'Camera: {self.device_id} NOT AVAILABLE')
            return

        if Core._pipeline is None:
            Core._pipeline = dai.Pipeline()
            setup_pipeline(Core._pipeline, self.model_path, self.fps, self.do_color, self.do_stereo, self.do_person, self.lowres, self.show_stereo, True)

        try:
            self.device = self._try_device(self.device_id, Core._pipeline, num_tries=3)
        except Exception as e:
            print(f'Could not open device: {e}')
            return

        self.frame_queue =      self.device.getOutputQueue(name='output_images', maxSize=4, blocking=False)
        self.tracklet_queue =   self.device.getOutputQueue(name='tracklets', maxSize=4, blocking=False)
        self.color_control =    self.device.getInputQueue(name='color_control')
        self.mono_control =     self.device.getInputQueue(name='mono_control')
        self.stereo_control =   self.device.getInputQueue(name='stereo_control')

        self.ex_video =         self.device.getInputQueue(name='ex_video')
        if self.do_stereo:
            self.ex_left =          self.device.getInputQueue(name='ex_left')
            self.ex_right =         self.device.getInputQueue(name='ex_right')

        self.frame_queue.addCallback(self._frame_callback)
        self.tracklet_queue.addCallback(self._tracker_callback)

        print(f'Camera: {self.device_id} OPEN')
        self.device_open: bool =        True

    def _close(self) -> None: # override

        if not self.device_open: return
        print(f'Camera: {self.device_id} CLOSING')
        self.ex_video.close()
        if self.do_stereo:
            self.ex_left.close()
            self.ex_right.close()
        super()._close()

    # def _frame_callback(self, message_group: dai.MessageGroup) -> None: # override
    #     return
    #     super()._frame_callback(message_group)

    # def _tracker_callback(self, msg: dai.RawTracklets) -> None: # override
    #     super()._tracker_callback(msg)

    def _video_frame_callback(self, id: int, frame_type: FrameType, frame: np.ndarray) -> None:

        if not self.device_open:
            return
        current_time: float = time.monotonic()
        height, width = frame.shape[:2]
        if id == self.id:
            if frame_type == FrameType.VIDEO:

                img = dai.ImgFrame()
                img.setType(dai.ImgFrame.Type.BGR888p)
                img.setData(self.to_planar(frame, (width, height)))
                img.setTimestamp(current_time) # type: ignore

                img.setWidth(width)
                img.setHeight(height)

                self.ex_video.send(img)

            if frame_type == FrameType.LEFT and self.do_stereo:
                img = dai.ImgFrame()
                img.setType(dai.ImgFrame.Type.RAW8)
                img.setInstanceNum(dai.CameraBoardSocket.CAM_A) # type: ignore
                img.setData(frame.flatten())
                img.setTimestamp(current_time) # type: ignore
                img.setWidth(width)
                img.setHeight(height)

                if self.do_stereo:
                    self.ex_left.send(img)

            if frame_type == FrameType.RIGHT:
                img = dai.ImgFrame()
                img.setType(dai.ImgFrame.Type.RAW8)
                img.setInstanceNum(dai.CameraBoardSocket.CAM_B) # type: ignore
                img.setData(frame.flatten())
                img.setTimestamp(current_time) # type: ignore
                img.setWidth(width)
                img.setHeight(height)
                if self.do_stereo:
                    self.ex_left.send(img)

    @staticmethod
    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        return resize(arr, shape).transpose(2, 0, 1).flatten()