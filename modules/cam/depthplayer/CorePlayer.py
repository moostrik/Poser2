
from modules.cam.depthcam.Core import *
from modules.cam.depthplayer.SyncPlayer import SyncPlayer
from cv2 import resize, COLOR_RGB2GRAY, cvtColor
from time import process_time
from datetime import timedelta

class CorePlayer(Core):

    def __init__(self, gui, syncplayer: SyncPlayer, device_id: str, model_path:str, fps: int = 30,
                 do_color: bool = True, do_stereo: bool = True, do_person: bool = True,
                 lowres: bool = False, show_stereo: bool = False) -> None:

        if do_stereo and not do_person:
            show_stereo = True  # stereo pipeline needs to be connected (in case of no person detection)

        super().__init__(gui, device_id, model_path, fps, do_color, do_stereo, do_person, lowres, show_stereo)

        self.sync_player: SyncPlayer = syncplayer
        self.ex_video:  dai.DataInputQueue
        self.ex_left:   dai.DataInputQueue
        self.ex_right:  dai.DataInputQueue


        self.passthrough: bool = True

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

        ex_video: dai.DataInputQueue =          self.device.getInputQueue(name='ex_video')
        self.inputs[input.VIDEO_FRAME_IN] =     ex_video
        video_queue: dai.DataOutputQueue =      self.device.getOutputQueue(name='video', maxSize=4, blocking=False)
        self.outputs[output.VIDEO_FRAME_OUT] =  video_queue
        video_queue.addCallback(self._video_callback)
        if self.do_stereo:
            stereo_control: dai.DataInputQueue =    self.device.getInputQueue('stereo_control')
            self.inputs[input.STEREO_CONTROL] =     stereo_control
            ex_left: dai.DataInputQueue =           self.device.getInputQueue(name='ex_left')
            self.inputs[input.LEFT_FRAME_IN] =      ex_left
            ex_right: dai.DataInputQueue =          self.device.getInputQueue(name='ex_right')
            self.inputs[input.RIGHT_FRAME_IN] =     ex_right
            left_queue: dai.DataOutputQueue =       self.device.getOutputQueue(name='left', maxSize=4, blocking=False)
            self.outputs[output.LEFT_FRAME_OUT] =   left_queue
            left_queue.addCallback(self._left_callback)
            right_queue: dai.DataOutputQueue =      self.device.getOutputQueue(name='right', maxSize=4, blocking=False)
            self.outputs[output.RIGHT_FRAME_OUT] =  right_queue
            right_queue.addCallback(self._right_callback)
            if self.show_stereo:
                stereo_queue: dai.DataOutputQueue =   self.device.getOutputQueue(name='stereo', maxSize=4, blocking=False)
                self.outputs[output.STEREO_FRAME_OUT] = stereo_queue
                stereo_queue.addCallback(self._stereo_callback)

        if self.do_person:
            self.tracklet_queue: dai.DataOutputQueue =   self.device.getOutputQueue(name='tracklets', maxSize=4, blocking=False)
            self.outputs[output.TRACKLETS_OUT] = self.tracklet_queue
            self.tracklet_queue.addCallback(self._tracker_callback)

        print(f'Camera: {self.device_id} OPEN')
        self.device_open: bool =        True

    def _video_frame_callback(self, id: int, frame_type: FrameType, frame: np.ndarray) -> None:
        if not self.device_open:
            return

        frame_time = timedelta(seconds = process_time())
        height, width = frame.shape[:2]
        if id == self.id:
            if frame_type == FrameType.VIDEO and input.VIDEO_FRAME_IN in self.inputs:
                img = dai.ImgFrame()
                # img.setInstanceNum(int(dai.CameraBoardSocket.CAM_A))
                if self.do_color:
                    img.setType(dai.ImgFrame.Type.BGR888p)
                    img.setData(self.to_planar(frame, (width, height)))
                else:
                    img.setType(dai.ImgFrame.Type.RAW8)
                    if frame.shape[2] == 3:
                        frame = cvtColor(frame, COLOR_RGB2GRAY)
                    img.setData(frame.flatten())
                img.setTimestamp(frame_time)
                img.setWidth(width)
                img.setHeight(height)
                self.inputs[input.VIDEO_FRAME_IN].send(img)

            if frame_type == FrameType.LEFT and input.LEFT_FRAME_IN in self.inputs:
                img = dai.ImgFrame()
                img.setType(dai.ImgFrame.Type.RAW8)
                img.setInstanceNum(int(dai.CameraBoardSocket.CAM_B))
                img.setData(frame.flatten())
                img.setTimestamp(frame_time)
                img.setWidth(width)
                img.setHeight(height)
                self.inputs[input.LEFT_FRAME_IN].send(img)

            if frame_type == FrameType.RIGHT and input.RIGHT_FRAME_IN in self.inputs:
                img = dai.ImgFrame()
                img.setType(dai.ImgFrame.Type.RAW8)
                img.setInstanceNum(int(dai.CameraBoardSocket.CAM_C))
                img.setData(frame.flatten())
                img.setTimestamp(frame_time)
                img.setWidth(width)
                img.setHeight(height)
                self.inputs[input.RIGHT_FRAME_IN].send(img)

    @staticmethod
    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        return resize(arr, shape).transpose(2, 0, 1).flatten()