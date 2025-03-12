# DOCS
# https://oak-web.readthedocs.io/
# https://docs.luxonis.com/software/depthai/examples/depth_post_processing/

import cv2
import numpy as np
import depthai as dai
from typing import Set
from PIL import Image, ImageOps

from modules.cam.DepthAiPipeline import SetupPipeline
from modules.cam.DepthAiDefines import *
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
    def __init__(self, modelPath:str, fps: int = 30, doColor: bool = True, doStereo: bool = True, doPerson: bool = True, lowres: bool = False, showLeft: bool = False) -> None:

        # FIXED SETTINGS
        self.modelpath: str =           modelPath
        self.fps: int =                 fps
        self.doColor: bool =            doColor
        self.doStereo: bool =           doStereo
        self.doPerson: bool =           doPerson
        self.lowres: bool =             lowres
        self.showLeft: bool =           False
        if self.doStereo and self.doColor and showLeft:
            self.showLeft: bool =       True

        # GENERAL SETTINGS
        self.previewType =              PreviewType.VIDEO
        self.flipH: bool =              False
        self.flipV: bool =              False

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
        self.dataQueue:                 dai.DataOutputQueue
        self.dataCallbackId:            int

        # CALLBACKS
        self.frameCallbacks: Set[FrameCallback] = set()
        self.detectionCallbacks: Set[DetectionCallback] = set()
        self.trackerCallbacks: Set[TrackerCallback] = set()

        # OTHER
        self.deviceOpen: bool =         False
        self.capturing:  bool =         False
        self.fps_counter =              FPS()

        self.errorFrame: np.ndarray =   np.zeros((720, 1280, 3), dtype=np.uint8)
        if self.lowres:
            self.errorFrame = cv2.resize(self.errorFrame, (640, 360))
        self.errorFrame[:,:,2] =        255

    def __exit__(self) -> None:
        self.close()

    def open(self) -> bool:
        if self.deviceOpen: return True

        pipeline = dai.Pipeline()
        self.stereoConfig = SetupPipeline(pipeline, self.modelpath, self.fps, self.doColor, self.doStereo, self.doPerson, self.lowres, self.showLeft)

        try: self.device = dai.Device(pipeline)
        except Exception as e:
            print('could not open camera, error', e, 'try again')
            try: self.device = dai.Device(pipeline)
            except Exception as e:
                print('still could not open camera, error', e)
                return False

        self.dataQueue =    self.device.getOutputQueue(name='output_images', maxSize=4, blocking=False)
        self.colorControl = self.device.getInputQueue('color_control')
        self.monoControl =  self.device.getInputQueue('mono_control')
        self.stereoControl =self.device.getInputQueue('stereo_control')

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
        self.dataQueue.close()

    def startCapture(self) -> None:
        if not self.deviceOpen:
            print('CamDepthAi:start', 'device is not open')
            return
        if self.capturing: return
        self.dataCallbackId = self.dataQueue.addCallback(self.updateData)

    def stopCapture(self) -> None:
        if not self.capturing: return
        self.dataQueue.removeCallback(self.dataCallbackId)

    def updateData(self, daiMessages) -> None:
        self.updateFPS()
        if self.previewType == PreviewType.NONE:
            return
        if len(self.frameCallbacks) == 0:
            return

        video_frame:  np.ndarray | None = None
        stereo_frame: np.ndarray | None = None
        mono_frame:   np.ndarray | None = None
        mask_frame:   np.ndarray | None = None
        masked_frame: np.ndarray | None = None
        detections:   Detections | None = None
        tracklets:    Tracklets  | None = None

        for name, msg in daiMessages:
            if name == 'video':
                video_frame = msg.getCvFrame() #type:ignore
                self.updateColorControl(msg)
            elif name == 'stereo':
                stereo_frame = self.updateStereo(msg.getCvFrame()) #type:ignore
                self.updateMonoControl(msg)
            elif name == 'mono':
                mono_frame = msg.getCvFrame() #type:ignore
            elif name == 'detection':
                detections = msg.detections
                self.numDetections = len(msg.detections)
            elif name == 'tracklets':
                tracklets = msg.tracklets
                self.numTracklets = len(msg.tracklets)
                pass
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
        if self.previewType == PreviewType.MONO and mono_frame is not None:
            return_frame = cv2.cvtColor(mono_frame, cv2.COLOR_GRAY2RGB)  # type: ignore
        if self.previewType == PreviewType.STEREO and stereo_frame is not None:
            return_frame = stereo_frame
        if self.previewType == PreviewType.MASK and mask_frame is not None:
            return_frame = cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2RGB)  # type: ignore
        if self.previewType == PreviewType.MASKED and masked_frame is not None:
            return_frame = masked_frame

        return_frame = self.flip(return_frame)

        for c in self.frameCallbacks:
            c(return_frame)

        if detections is not None:
            for c in self.detectionCallbacks:
                c(detections)
        if tracklets is not None:
            for c in self.trackerCallbacks:
                c(tracklets)

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

    def flip(self, frame: np.ndarray) -> np.ndarray:
        if self.flipH and self.flipV:
            return cv2.flip(frame, -1)
        if self.flipH:
            return cv2.flip(frame, 1)
        if self.flipV:
            return cv2.flip(frame, 0)
        return frame

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

    def addDetectionCallback(self, callback: DetectionCallback) -> None:
        self.detectionCallbacks.add(callback)
    def discardDetectionCallback(self, callback: DetectionCallback) -> None:
        self.detectionCallbacks.discard(callback)
    def clearDetectionCallbacks(self) -> None:
        self.detectionCallbacks.clear()

    # FPS
    def updateFPS(self) -> None:
        self.fps_counter.processed()
    def getFPS(self) -> float:
        return self.fps_counter.get_rate_average()













