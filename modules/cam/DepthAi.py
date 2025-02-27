import cv2
import numpy as np
import depthai as dai
from threading import Lock
from datetime import timedelta
import time
from PIL import Image, ImageOps
from enum import Enum

class PreviewType(Enum):
    NONE =  0
    VIDEO = 1
    MONO =  2
    STEREO= 3
    MASK =  4
    MASKED= 5

PreviewTypeNames: list[str] = [e.name for e in PreviewType]

exposureRange: tuple[int, int] =    (1000, 33000)
isoRange: tuple[int, int] =         ( 100, 1600 )
focusRange: tuple[int, int] =       (   0, 255  )
whiteBalanceRange: tuple[int, int]= (1000, 12000)

depthDecimationRange: tuple[int, int]=  (   0, 4    )
depthSpeckleRange: tuple[int, int]=     (   0, 50   )
depthHoleFillingRange: tuple[int, int]= (   0, 4    )
depthHoleIterRange: tuple[int, int]=    (   1, 4    )
depthTempPersistRange: tuple[int, int]= (   0, 8    )
depthTresholdRange: tuple[int, int]=    ( 500, 15000)
depthDisparityRange: tuple[int, int]=   (   0, 48   )

def clamp(num: int | float, size: tuple[int | float, int | float]) -> int | float:
    return max(size[0], min(num, size[1]))

def fit(image: np.ndarray, width, height) -> np.ndarray:
    h, w = image.shape[:2]
    if w == width and h == height:
        # print('yes', w, h, width, height)
        return image
    pil_image = Image.fromarray(image)
    size = (width, height)
    fit_image = ImageOps.fit(pil_image, size)
    return np.asarray(fit_image)

def makeWarpList(flipH: bool, flipV:bool, rotate90:bool, zoom: float, offset: tuple[float,float], perspective: tuple[float,float]) -> list[dai.Point2f]:
    z:float = (zoom-1.0) * 0.25
    Lz:float = 0.0+z
    Hz:float = 1.0-z
    Ox: float = offset[0] * z
    Oy: float = offset[1] * z
    W = np.array([[Lz+Ox,Lz+Oy],[Hz+Ox,Lz+Oy],[Hz+Ox,Hz+Oy],[Lz+Ox,Hz+Oy]])
    P = np.array([[-perspective[0],-perspective[1]],[perspective[0],perspective[1]],[-perspective[0],-perspective[1]],[perspective[0],perspective[1]]])
    W += P
    if (rotate90):
        W = [W[3],W[0],W[1],W[2]]
    if flipH:
        W = [W[3],W[2],W[1],W[0]]
    if flipV:
        W = [W[1],W[0],W[3],W[2]]
    dW: list[dai.Point2f] = [dai.Point2f(W[0][0], W[0][1]), dai.Point2f(W[1][0], W[1][1]),dai.Point2f(W[2][0], W[2][1]),dai.Point2f(W[3][0], W[3][1])]
    return dW

def setupStereo(pipeline : dai.Pipeline, addMonoOutput:bool = False) -> None:
    monoLeft: dai.node.MonoCamera = pipeline.create(dai.node.MonoCamera)
    monoRight: dai.node.MonoCamera = pipeline.create(dai.node.MonoCamera)
    color: dai.node.ColorCamera = pipeline.create(dai.node.ColorCamera)
    stereo: dai.node.StereoDepth = pipeline.create(dai.node.StereoDepth)
    sync: dai.node.Sync = pipeline.create(dai.node.Sync)

    colorControl: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
    colorControl.setStreamName('color_control')
    colorControl.out.link(color.inputControl)

    stereoControl: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
    stereoControl.setStreamName('stereo_control')
    stereoControl.out.link(stereo.inputConfig)

    outputImages: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
    outputImages.setStreamName("output_images")

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    monoLeft.setCamera("left")
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    monoRight.setCamera("right")

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(False)

    color.setCamera("color")
    color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)

    sync.setSyncThreshold(timedelta(milliseconds=50))

    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    stereo.disparity.link(sync.inputs["stereo"])
    color.video.link(sync.inputs["video"])
    if addMonoOutput:
        monoLeft.out.link(sync.inputs["mono"])

    sync.out.link(outputImages.input)

class DepthAi():
    def __init__(self, doMono: bool = True) -> None:

        self.previewType: PreviewType = PreviewType.VIDEO

        self.doMono: bool = doMono

        self.frameCallbacks: set = set()

        self.outWidth: int = 1280
        self.outHeight: int = 720
        self.flipH: bool = False
        self.flipV: bool = False
        self.rotate90: bool = False
        self.errorFrame:   np.ndarray = np.zeros((self.outHeight, self.outWidth, 3), dtype=np.uint8)
        self.errorFrame[:,:,2] = 255

        # COLOR SETTINGS
        self.autoExposure: bool     = True
        self.autoFocus: bool        = True
        self.autoWhiteBalance: bool = True
        self.exposure: int          = 0
        self.iso: int               = 0
        self.focus: int             = 0
        self.whiteBalance: int      = 0

        # DEPTH SETTINGS
        self.depthDecimation: int   = depthDecimationRange[0]
        self.depthSpeckle: int      = depthSpeckleRange[0]
        self.depthHoleFilling: int  = depthHoleFillingRange[0]
        self.depthHoleIter: int     = depthHoleIterRange[0]
        self.depthTempPersist: int  = depthTempPersistRange[0]
        self.depthTresholdMin: int  = depthTresholdRange[0]
        self.depthTresholdMax: int  = depthTresholdRange[1]
        self.depthDispShift: int    = depthDisparityRange[0]

        # DAI
        self.device:        dai.Device
        self.dataQueue:     dai.DataOutputQueue
        self.colorControl:  dai.DataInputQueue
        self.stereoControl: dai.DataInputQueue

        self.deviceOpen: bool = False
        self.capturing:  bool = False

    def __exit__(self) -> None:
        self.close()

    def open(self) -> bool:
        if self.deviceOpen: return True

        pipeline = dai.Pipeline()
        setupStereo(pipeline, self.doMono)

        try: self.device = dai.Device(pipeline)
        except Exception as e:
            print('could not open camera, error', e, 'try again')
            try: self.device = dai.Device(pipeline)
            except Exception as e:
                print('still could not open camera, error', e)
                return False

        self.dataQueue =    self.device.getOutputQueue(name='output_images', maxSize=4, blocking=False)
        self.colorControl = self.device.getInputQueue('color_control')
        self.stereoControl =self.device.getInputQueue('stereo_control')

        self.deviceOpen = True
        return True

    def close(self) -> None:
        if not self.deviceOpen: return
        if self.capturing: self.stopCapture()
        self.deviceOpen = False

        self.device.close()
        self.stereoControl.close()
        self.colorControl.close()
        self.dataQueue.close()

    def startCapture(self) -> None:
        if not self.deviceOpen:
            print('CamDepthAi:start', 'device is not open')
            return
        if self.capturing: return
        self.updateWarp
        self.colorId: int = self.dataQueue.addCallback(self.updateData)

    def stopCapture(self) -> None:
        if not self.capturing: return
        self.dataQueue.removeCallback(self.colorId)

    def updateData(self, daiMessages) -> None:
        if self.previewType == PreviewType.NONE:
            return
        if len(self.frameCallbacks) == 0:
            return

        video_frame:  np.ndarray | None = None
        stereo_frame: np.ndarray | None = None
        mono_frame:   np.ndarray | None = None
        mask_frame:   np.ndarray | None = None
        masked_frame: np.ndarray | None = None

        for name, msg in daiMessages:
            if name == 'video':
                video_frame = msg.getCvFrame() #type:ignore
                self.updateControlValues(msg)
            elif name == 'stereo':
                stereo_frame = self.updateStereo(msg.getCvFrame()) #type:ignore
            elif name == 'mono':
                mono_frame = msg.getCvFrame() #type:ignore
            else:
                print('unknown message', name)

        if stereo_frame is not None:
            mask_frame = self.updateMask(stereo_frame)

        if video_frame is not None and mask_frame is not None:
            masked_frame = self.applyMask(video_frame, mask_frame)

        return_frame: np.ndarray | None = None
        if self.previewType == PreviewType.VIDEO:
            return_frame = video_frame
        if self.previewType == PreviewType.MONO:
            return_frame = mono_frame
        if self.previewType == PreviewType.STEREO:
            return_frame = stereo_frame
        if self.previewType == PreviewType.MASK:
            return_frame = mask_frame
        if self.previewType == PreviewType.MASKED:
            return_frame = masked_frame
        if return_frame is None:
            return_frame = self.errorFrame

        return_frame = self.flip(return_frame)

        for c in self.frameCallbacks:
            c(return_frame)

    def updateStereo(self, frame: np.ndarray) -> np.ndarray:
        return (frame * (255 / 95)).astype(np.uint8)

    def updateMask(self, frame: np.ndarray) -> np.ndarray:
        _, binary_mask = cv2.threshold(frame, 10, 255, cv2.THRESH_BINARY)
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

    def updateControlValues(self, frame) -> None:
        if (self.autoExposure):
            self.exposure = frame.getExposureTime().total_seconds()*1000000
            self.iso = frame.getSensitivity()
        if (self.autoFocus):
            self.focus = frame.getLensPosition()
        if (self.autoWhiteBalance):
            self.whiteBalance = frame.getColorTemperature()

    def iscapturing(self) ->bool:
        return self.capturing

    def isOpen(self) -> bool:
        return self.deviceOpen


    def setPreview(self, value: PreviewType | int | str) -> None:
        if isinstance(value, str) and value in PreviewTypeNames:
            self.previewType = PreviewType(PreviewTypeNames.index(value))
        else:
            self.previewType = PreviewType(value)

    def setAutoExposure(self, value) -> None:
        if not self.deviceOpen: return
        self.autoExposure = value
        if value == False:
            self.setExposureIso(self.exposure, self.iso)
            return
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureEnable()
        self.colorControl.send(ctrl)

    def setAutoFocus(self, value) -> None:
        if not self.deviceOpen: return
        self.autoFocus = value
        if value == False:
            self.setFocus(self.focus)
            return
        ctrl = dai.CameraControl()
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
        self.colorControl.send(ctrl)

    def setAutoWhiteBalance(self, value) -> None:
        if not self.deviceOpen: return
        self.autoWhiteBalance = value
        if value == False:
            self.setWhiteBalance(self.whiteBalance)
            return
        ctrl = dai.CameraControl()
        ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
        self.colorControl.send(ctrl)

    def setExposureIso(self, exposure: int, iso: int) -> None:
        if not self.deviceOpen: return
        self.autoExposure = False
        self.exposure = int(clamp(exposure, exposureRange))
        self.iso = int(clamp(iso, isoRange))
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(int(self.exposure), int(self.iso))
        self.colorControl.send(ctrl)

    def setExposure(self, value : int) -> None:
        self.setExposureIso(value, self.iso)

    def setIso(self, value: int) -> None:
        self.setExposureIso(self.exposure, value)

    def setFocus(self, value: int) -> None:
        if not self.deviceOpen: return
        self.autoFocus = False
        ctrl = dai.CameraControl()
        self.focus = int(clamp(value, focusRange))
        ctrl.setManualFocus(int(self.focus))
        self.colorControl.send(ctrl)

    def setWhiteBalance(self, value: int) -> None:
        if not self.deviceOpen: return
        self.autoWhiteBalance = False
        ctrl = dai.CameraControl()
        self.whiteBalance = int(clamp(value, whiteBalanceRange))
        ctrl.setManualWhiteBalance(int(self.whiteBalance))
        self.colorControl.send(ctrl)


    def updateDepthControl(self) -> None:
        if not self.deviceOpen: return
        cnfg = dai.RawStereoDepthConfig()
        cnfg.postProcessing.decimationFilter.decimationFactor = self.depthDecimation
        cnfg.postProcessing.speckleFilter.enable = self.depthSpeckle > 0
        cnfg.postProcessing.speckleFilter.speckleRange = self.depthSpeckle
        cnfg.postProcessing.spatialFilter.enable = self.depthHoleFilling >0
        cnfg.postProcessing.spatialFilter.holeFillingRadius = self.depthHoleFilling
        cnfg.postProcessing.spatialFilter.numIterations = self.depthHoleIter
        cnfg.postProcessing.thresholdFilter.minRange = self.depthTresholdMin
        cnfg.postProcessing.thresholdFilter.maxRange = self.depthTresholdMax
        cnfg.postProcessing.temporalFilter.enable = self.depthTempPersist > 0
        cnfg.postProcessing.temporalFilter.persistencyMode = cnfg.postProcessing.temporalFilter.persistencyMode.PERSISTENCY_OFF
        if   self.depthTempPersist == 1: cnfg.postProcessing.temporalFilter.persistencyMode = cnfg.postProcessing.temporalFilter.persistencyMode.VALID_1_IN_LAST_2
        elif self.depthTempPersist == 2: cnfg.postProcessing.temporalFilter.persistencyMode = cnfg.postProcessing.temporalFilter.persistencyMode.VALID_1_IN_LAST_5
        elif self.depthTempPersist == 3: cnfg.postProcessing.temporalFilter.persistencyMode = cnfg.postProcessing.temporalFilter.persistencyMode.VALID_1_IN_LAST_8
        elif self.depthTempPersist == 4: cnfg.postProcessing.temporalFilter.persistencyMode = cnfg.postProcessing.temporalFilter.persistencyMode.VALID_2_IN_LAST_3
        elif self.depthTempPersist == 5: cnfg.postProcessing.temporalFilter.persistencyMode = cnfg.postProcessing.temporalFilter.persistencyMode.VALID_2_IN_LAST_4
        elif self.depthTempPersist == 6: cnfg.postProcessing.temporalFilter.persistencyMode = cnfg.postProcessing.temporalFilter.persistencyMode.VALID_2_OUT_OF_8
        elif self.depthTempPersist == 7: cnfg.postProcessing.temporalFilter.persistencyMode = cnfg.postProcessing.temporalFilter.persistencyMode.VALID_8_OUT_OF_8

        cnfg.algorithmControl.disparityShift = self.depthDispShift

        self.stereoControl.send(cnfg)

    def setDepthDecimation(self, value: int) -> None:
        self.depthDecimation = int(clamp(value, depthDecimationRange))
        self.updateDepthControl()

    def setDepthSpeckle(self, value: int) -> None:
        self.depthSpeckle = int(clamp(value, depthSpeckleRange))
        self.updateDepthControl()

    def setDepthHoleFilling(self, value: int) -> None:
        self.depthHoleFilling = int(clamp(value, depthHoleFillingRange))
        self.updateDepthControl()

    def setDepthHoleIter(self, value: int) -> None:
        self.depthHoleIter = int(clamp(value, depthHoleIterRange))
        self.updateDepthControl()

    def setDepthTempPersist(self, value: int) -> None:
        self.depthTempPersist = int(clamp(value, depthTempPersistRange))
        self.updateDepthControl()

    def setDepthTresholdMin(self, value: int) -> None:
        self.depthTresholdMin = int(clamp(value, depthTresholdRange))
        self.updateDepthControl()

    def setDepthTresholdMax(self, value: int) -> None:
        self.depthTresholdMax = int(clamp(value, depthTresholdRange))
        self.updateDepthControl()

    def setDepthDisparityShift(self, value: int) -> None:
        self.depthDispShift = int(clamp(value, depthDisparityRange))
        self.updateDepthControl()


    def updateWarp(self) -> None:
        pass

    def setFlipH(self, flipH: bool) -> None:
        self.flipH = flipH
        self.updateWarp()

    def setFlipV(self, flipV: bool) -> None:
        self.flipV = flipV
        self.updateWarp()

    def setRotate90(self, rotate90: bool) -> None:
        self.rotate90 = rotate90
        self.updateWarp()

    def setZoom(self, zoom:float) -> None:
        self.zoom = zoom
        self.updateWarp()


    def addFrameCallback(self, callback) -> None:
        self.frameCallbacks.add(callback)

    def clearFrameCallbacks(self) -> None:
        self.frameCallbacks = set()

















