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

        # GENERAL SETTINGS
        self.previewType =              FrameType.VIDEO

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
        self.numTracklets: int =        0
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

        # OTHER
        self.deviceOpen: bool =         False
        self.capturing:  bool =         False
        self.fps_counter =              FPS(120)
        self.tps_counter =              FPS(120)

        self.errorFrame: np.ndarray =   np.zeros((720, 1280, 3), dtype=np.uint8)
        if self.lowres:
            self.errorFrame = cv2.resize(self.errorFrame, (640, 360))
        self.errorFrame[:,:,2] =        255

        self.frame_types: set[FrameType] = set()
        if self.doColor: self.frame_types.add(FrameType.VIDEO)
        else: self.frame_types.add(FrameType.LEFT)
        if self.doStereo:
            self.frame_types.add(FrameType.LEFT)
            self.frame_types.add(FrameType.RIGHT)

        # CALLBACKS
        self.previewCallbacks: Set[PreviewCallback] = set()
        self.trackerCallbacks: Set[TrackerCallback] = set()
        self.fpsCallbacks: Set[FPSCallback] = set()
        self.frameCallbacks: dict[FrameType, Set[FrameCallback]] = {}
        for t in self.frame_types:
            if t == FrameType.NONE: continue
            self.frameCallbacks[t] = set()

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

        self.frameCallbacks.clear()
        self.previewCallbacks.clear()
        self.trackerCallbacks.clear()

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

    def updateFrames(self, messageGroup: dai.MessageGroup) -> None:
        self.updateFPS()

        for name, msg in messageGroup:
            if name == 'video':
                self.updateColorControl(msg)
                frame: np.ndarray = msg.getCvFrame() #type:ignore
                self.updateCallbacks(FrameType.VIDEO, frame)

            elif name == 'left':
                self.updateMonoControl(msg)
                frame: np.ndarray = msg.getCvFrame() #type:ignore
                self.updateCallbacks(FrameType.LEFT, frame)

            elif name == 'right':
                frame = msg.getCvFrame() #type:ignore
                self.updateCallbacks(FrameType.RIGHT, frame)

            elif name == 'stereo':
                frame = self.updateStereo(msg.getCvFrame()) #type:ignore
                frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                self.updateCallbacks(FrameType.STEREO, frame)

            else:
                print('unknown message', name)

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

    def updateCallbacks(self, frameType: FrameType, frame: np.ndarray) -> None:
        for c in self.frameCallbacks[frameType]:
            c(self.ID, frameType, frame)
        if self.previewType == frameType:
            for c in self.previewCallbacks:
                c(self.ID, frame)

    # CALLBACKS
    def addFrameCallback(self, frameType: FrameType, callback: FrameCallback) -> None:
        self.frameCallbacks[frameType].add(callback)
    def discardFrameCallback(self, frameType: FrameType, callback: FrameCallback) -> None:
        self.frameCallbacks[frameType].discard(callback)
    def clearFrameCallbacks(self) -> None:
        self.frameCallbacks.clear()

    def addPreviewCallback(self, callback: PreviewCallback) -> None:
        self.previewCallbacks.add(callback)
    def discardPreviewCallback(self, callback: PreviewCallback) -> None:
        self.previewCallbacks.discard(callback)
    def clearPreviewCallbacks(self) -> None:
        self.previewCallbacks.clear()

    def addTrackerCallback(self, callback: TrackerCallback) -> None:
        self.trackerCallbacks.add(callback)
    def discardTrackerCallback(self, callback: TrackerCallback) -> None:
        self.trackerCallbacks.discard(callback)
    def clearTrackerCallbacks(self) -> None:
        self.trackerCallbacks.clear()

    def addFPSCallback(self, callback: FPSCallback) -> None:
        self.fpsCallbacks.add(callback)
    def discardFPSCallback(self, callback: FPSCallback) -> None:
        self.fpsCallbacks.discard(callback)
    def clearFPSCallbacks(self) -> None:
        self.fpsCallbacks.clear()

    # FPS
    def updateFPS(self) -> None:
        self.fps_counter.processed()
        for c in self.fpsCallbacks:
            c(self.ID, self.fps_counter.get_rate_average())
    def getFPS(self) -> float:
        return self.fps_counter.get_rate_average()
    def updateTPS(self) -> None:
        self.tps_counter.processed()
    def getTPS(self) -> float:
        return self.tps_counter.get_rate_average()













