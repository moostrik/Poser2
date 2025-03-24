# DOCS
# https://oak-web.readthedocs.io/
# https://docs.luxonis.com/software/depthai/examples/depth_post_processing/

import cv2
import numpy as np
import depthai as dai
from typing import Set
from PIL import Image, ImageOps

from modules.cam.DepthAi.Pipeline import SetupPipeline
from modules.cam.DepthAi.Definitions import *
from modules.utils.FPS import FPS


def clamp(num: int | float, size: tuple[int | float, int | float]) -> int | float:
    return max(size[0], min(num, size[1]))

def fit(image: np.ndarray, width, height) -> np.ndarray:
    h, w = image.shape[:2]
    if w == width and h == height:
        return image
    pil_image: Image.Image = Image.fromarray(image)
    size: tuple[int, int] = (width, height)
    fit_image: Image.Image = ImageOps.fit(pil_image, size)
    return np.asarray(fit_image)

class DepthAiCore():
    _id_counter = 0
    def __init__(self, modelPath:str, fps: int = 30, doColor: bool = True, doStereo: bool = True, doPerson: bool = True, lowres: bool = False, showStereo: bool = False) -> None:
        self.ID: int =                  DepthAiCore._id_counter
        self.IDs: str =                 str(self.ID)
        DepthAiCore._id_counter +=      1

        # FIXED SETTINGS
        self.modelpath: str =           modelPath
        self.fps: int =                 fps
        self.doColor: bool =            doColor
        self.doStereo: bool =           doStereo
        self.doPerson: bool =           doPerson
        self.lowres: bool =             lowres
        self.showStereo: bool =         showStereo

        self.preview_types: list[PreviewType] = []
        self.preview_types.append(PreviewType.NONE)
        if self.doColor: self.preview_types.append(PreviewType.VIDEO)
        if self.doStereo:
            self.preview_types.append(PreviewType.LEFT)
            self.preview_types.append(PreviewType.RIGHT)
            if self.showStereo:
                self.preview_types.append(PreviewType.STEREO)

        # GENERAL SETTINGS
        self.previewType =              PreviewType.VIDEO

        # COLOR SETTINGS
        self.colorAutoExposure: bool =  True
        self.colorAutoFocus: bool =     True
        self.colorAutoBalance: bool =   True
        self.colorExposure: int =       0
        self.colorIso: int =            0
        self.colorFocus: int =          0
        self.colorBalance: int =        0
        self.colorContrast: int =       0
        self.colorBrightness: int =     0
        self.colorLumaDenoise: int =    0
        self.colorSaturation: int =     0
        self.colorSharpness: int =      0

        # MONO SETTINGS
        self.monoAutoExposure: bool =   True
        self.monoAutoFocus: bool =      True
        self.monoExposure: int =        0
        self.monoIso: int =             0

        # STEREO SETTINGS
        self.stereoConfig: dai.RawStereoDepthConfig = dai.RawStereoDepthConfig()

        # MASK SETTINGS
        self.depthTresholdMin:  int =   0
        self.depthTresholdMax:  int =   255

        # TRACKER SETTINGS
        self.numTracklets: int =           0
        self.numDetections: int =       0

        # DAI
        self.device:                    dai.Device
        self.colorControl:              dai.DataInputQueue
        self.monoControl:               dai.DataInputQueue
        self.stereoControl:             dai.DataInputQueue
        self.frameQueue:                dai.DataOutputQueue
        self.frameCallbackId:           int
        self.trackletQueue:             dai.DataOutputQueue
        self.trackletCallbackId:        int

        # CALLBACKS
        self.frameCallbacks: Set[FrameCallback] = set()
        self.trackerCallbacks: Set[TrackerCallback] = set()

        # OTHER
        self.deviceOpen: bool =         False
        self.capturing:  bool =         False
        self.fps_counter =              FPS()
        self.tps_counter =              FPS()

        self.errorFrame: np.ndarray =   np.zeros((720, 1280, 3), dtype=np.uint8)
        if self.lowres:
            self.errorFrame = cv2.resize(self.errorFrame, (640, 360))
        self.errorFrame[:,:,2] =        255

    def __exit__(self) -> None:
        self.close()

    def open(self) -> bool:
        if self.deviceOpen: return True

        pipeline = dai.Pipeline()
        self.stereoConfig = SetupPipeline(pipeline, self.modelpath, self.fps, self.doColor, self.doStereo, self.doPerson, self.lowres, self.showStereo)

        try: self.device = dai.Device(pipeline)
        except Exception as e:
            print('could not open camera, error', e, 'try again')
            try: self.device = dai.Device(pipeline)
            except Exception as e:
                print('still could not open camera, error', e)
                return False

        self.frameQueue =       self.device.getOutputQueue(name='output_images', maxSize=4, blocking=False)
        self.trackletQueue =    self.device.getOutputQueue(name='tracklets', maxSize=4, blocking=False)
        self.colorControl =     self.device.getInputQueue('color_control')
        self.monoControl =      self.device.getInputQueue('mono_control')
        self.stereoControl =    self.device.getInputQueue('stereo_control')

        self.deviceOpen = True
        return True

    def close(self) -> None:
        if not self.deviceOpen: return
        if self.capturing: self.stopCapture()
        self.deviceOpen = False

        self.device.close()
        self.stereoControl.close()
        self.monoControl.close()
        self.colorControl.close()
        self.frameQueue.close()
        self.trackletQueue.close()

    def startCapture(self) -> None:
        if not self.deviceOpen:
            print('CamDepthAi:start', 'device is not open')
            return
        if self.capturing: return
        self.frameCallbackId = self.frameQueue.addCallback(self.updateFrames)
        self.trackletCallbackId = self.trackletQueue.addCallback(self.updateTracker)

    def stopCapture(self) -> None:
        if not self.capturing: return
        self.frameQueue.removeCallback(self.frameCallbackId)
        self.trackletQueue.removeCallback(self.trackletCallbackId)

    def updateFrames(self, daiMessages) -> None:
        self.updateFPS()
        if self.previewType == PreviewType.NONE:
            return
        if len(self.frameCallbacks) == 0:
            return

        video_frame:  np.ndarray | None = None
        stereo_frame: np.ndarray | None = None
        left_frame:   np.ndarray | None = None
        right_frame:  np.ndarray | None = None
        mask_frame:   np.ndarray | None = None
        masked_frame: np.ndarray | None = None


        for name, msg in daiMessages:
            if name == 'video':
                video_frame = msg.getCvFrame() #type:ignore
                self.updateColorControl(msg)
            elif name == 'stereo':
                stereo_frame = self.updateStereo(msg.getCvFrame()) #type:ignore
                self.updateMonoControl(msg)
            elif name == 'left':
                left_frame = msg.getCvFrame() #type:ignore
            elif name == 'right':
                right_frame = msg.getCvFrame() #type:ignore
            else:
                print('unknown message', name)

        if stereo_frame is not None:
            mask_frame = self.updateMask(stereo_frame)
            stereo_frame = cv2.applyColorMap(stereo_frame, cv2.COLORMAP_JET)

        if video_frame is not None and mask_frame is not None:
            masked_frame = self.applyMask(video_frame, mask_frame)

        return_frame: np.ndarray = self.errorFrame
        if self.previewType == PreviewType.VIDEO and video_frame is not None:
            return_frame = video_frame
        if self.previewType == PreviewType.LEFT and left_frame is not None:
            # return_frame = cv2.cvtColor(left_frame, cv2.COLOR_GRAY2RGB)
            return_frame = left_frame
        if self.previewType == PreviewType.RIGHT and right_frame is not None:
            # return_frame = cv2.cvtColor(right_frame, cv2.COLOR_GRAY2RGB)
            return_frame = right_frame
        if self.previewType == PreviewType.STEREO and stereo_frame is not None:
            return_frame = stereo_frame

        for c in self.frameCallbacks:
            c(self.ID, return_frame)

    def updateTracker(self, msg) -> None:
        self.updateTPS()
        Ts = msg.tracklets
        self.numTracklets = len(Ts)
        for t in Ts:
            # tracklet: Tracklet = Tracklet.from_dai(t, self.ID)
            for c in self.trackerCallbacks:
                c(self.ID, t)

    def updateStereo(self, frame: np.ndarray) -> np.ndarray:
        return (frame * (255 / 95)).astype(np.uint8)

    def updateMask(self, frame: np.ndarray) -> np.ndarray:
        min: int = self.depthTresholdMin
        max: int = self.depthTresholdMax
        _, binary_mask = cv2.threshold(frame, min, max, cv2.THRESH_BINARY)
        return binary_mask

    def applyMask(self, color: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # resize color to mask size
        color = fit(color, mask.shape[1], mask.shape[0])

        return cv2.bitwise_and(color, color, mask=mask)

    def iscapturing(self) ->bool:
        return self.capturing

    def isOpen(self) -> bool:
        return self.deviceOpen

    def updateColorControl(self, frame) -> None:
        if (self.colorAutoExposure):
            self.colorExposure = frame.getExposureTime().total_seconds() * 1000000
            self.colorIso = frame.getSensitivity()
        if (self.colorAutoFocus):
            self.colorFocus = frame.getLensPosition()
        if (self.colorAutoBalance):
            self.colorBalance = frame.getColorTemperature()

    def updateMonoControl(self, frame) -> None:
        if (self.monoAutoExposure):
            self.monoExposure = frame.getExposureTime().total_seconds() * 1000000
            self.monoIso = frame.getSensitivity()

    # CALLBACKS
    def addFrameCallback(self, callback: FrameCallback) -> None:
        self.frameCallbacks.add(callback)
    def discardFrameCallback(self, callback: FrameCallback) -> None:
        self.frameCallbacks.discard(callback)
    def clearFrameCallbacks(self) -> None:
        self.frameCallbacks.clear()

    def addTrackerCallback(self, callback: TrackerCallback) -> None:
        self.trackerCallbacks.add(callback)
    def discardTrackerCallback(self, callback: TrackerCallback) -> None:
        self.trackerCallbacks.discard(callback)
    def clearTrackerCallbacks(self) -> None:
        self.trackerCallbacks.clear()

    # FPS
    def updateFPS(self) -> None:
        self.fps_counter.processed()
    def getFPS(self) -> float:
        return self.fps_counter.get_rate_average()
    def updateTPS(self) -> None:
        self.tps_counter.processed()
    def getTPS(self) -> float:
        return self.tps_counter.get_rate_average()













