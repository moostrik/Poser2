
from depthai import Pipeline
from modules.cam.depthcam.Core import *
from modules.cam.depthplayer.SyncPlayer import SyncPlayer
from cv2 import resize, COLOR_RGB2GRAY, cvtColor
from time import process_time
from datetime import timedelta

class CorePlayer(Core):

    def __init__(self, gui, syncplayer: SyncPlayer, device_id: str, settings:GeneralSettings) -> None:

        if settings.stereo and not settings.person:
            show_stereo = True  # stereo pipeline needs to be connected (in case of no person detection)

        super().__init__(gui, device_id, settings)

        self.sync_player: SyncPlayer = syncplayer
        self.ex_video:  dai.DataInputQueue
        self.ex_left:   dai.DataInputQueue
        self.ex_right:  dai.DataInputQueue

        self.passthrough: bool = settings.passthrough

    def start(self) -> None: # override
        if self.passthrough:
            self.sync_player.addFrameCallback(self._passthrough_frame_callback)
        else:
            self.sync_player.addFrameCallback(self._video_frame_callback)
        super().start()

    def stop(self) -> None: # override
        if self.passthrough:
            self.sync_player.discardFrameCallback(self._passthrough_frame_callback)
        else:
            self.sync_player.discardFrameCallback(self._video_frame_callback)
        super().stop()

    def run(self) -> None: # override
        if self.passthrough:
            self.stop_event.wait()
        else:
            super().run()

    def _setup_pipeline(self, pipeline: Pipeline) -> None: # override
        setup_pipeline(pipeline, self.model_path, self.fps, self.do_color, self.do_stereo, self.do_person, self.lowres, self.show_stereo, simulate=True)

    def _setup_queues(self) -> None: # override
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

    def _passthrough_frame_callback(self, id: int, frame_type: FrameType, frame: np.ndarray) -> None:
        if id != self.id:
            return
        self._update_callbacks(frame_type, frame)

    @staticmethod
    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        return resize(arr, shape).transpose(2, 0, 1).flatten()