import cv2
import numpy as np
import depthai as dai
from threading import Lock
import time
from PIL import Image, ImageOps

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

def setupColor(pipeline : dai.Pipeline, width: int, height: int) -> None:
    camColor = pipeline.create(dai.node.ColorCamera)
    camColor.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    # camColor.setVideoSize(int(width), int(height))
    camColor.setInterleaved(False)
    camColor.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    camColor.setPreviewSize(camColor.getVideoWidth(), camColor.getVideoHeight())

    colorControl = pipeline.create(dai.node.XLinkIn)
    colorControl.setStreamName('color_control')
    colorControl.out.link(camColor.inputControl)

    colorManip = pipeline.create(dai.node.ImageManip)
    frame_size= max(camColor.getVideoWidth() * camColor.getVideoHeight() * 3, 1920 * 1080 * 3)
    colorManip.setMaxOutputFrameSize(frame_size)
    colorManip.setNumFramesPool(1)
    colorManip.initialConfig.setCropRect(0.3, 0, 0.7, 1)
    # colorManip.initialConfig.setResize(width, height)
    camColor.preview.link(colorManip.inputImage)

    colorConfig = pipeline.create(dai.node.XLinkIn)
    colorConfig.setStreamName('color_manip_config')
    colorConfig.out.link(colorManip.inputConfig)

    colorImage = pipeline.create(dai.node.XLinkOut)
    colorImage.setStreamName('color_image')
    colorManip.out.link(colorImage.input)
    # camColor.preview.link(colorImage.input)

def setupDepth(pipeline : dai.Pipeline, width: int, height: int) -> None:
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight = pipeline.create(dai.node.MonoCamera)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)

    monoControl = pipeline.create(dai.node.XLinkIn)
    monoControl.setStreamName('mono_control')
    monoControl.out.link(monoRight.inputControl)
    monoControl.out.link(monoLeft.inputControl)

    depth = pipeline.create(dai.node.StereoDepth)
    depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    depth.setLeftRightCheck(True)
    depth.setExtendedDisparity(False)
    depth.setSubpixel(False)
    monoLeft.out.link(depth.left)
    monoRight.out.link(depth.right)

    depthConfig = pipeline.create(dai.node.XLinkIn)
    depthConfig.setStreamName('depth_config')
    depthConfig.out.link(depth.inputConfig)

    disparityManip = pipeline.create(dai.node.ImageManip)
    disparityManip.setMaxOutputFrameSize(width*height)
    disparityManip.initialConfig.setCropRect(0, 0, 1, 1)
    disparityManip.initialConfig.setResize(width, height)
    depth.disparity.link(disparityManip.inputImage)

    disparityConfig = pipeline.create(dai.node.XLinkIn)
    disparityConfig.setStreamName('disparity_manip_config')
    disparityConfig.out.link(disparityManip.inputConfig)

    disparity = pipeline.create(dai.node.XLinkOut)
    disparity.setStreamName('disparity')
    disparityManip.out.link(disparity.input)

def fit(image: np.ndarray, width, height) -> np.ndarray:
    h, w = image.shape[:2]
    if w == width and h == height:
        # print('yes', w, h, width, height)
        return image
    pil_image = Image.fromarray(image)
    size = (width, height)
    fit_image = ImageOps.fit(pil_image, size)
    return np.asarray(fit_image)

class DepthAi():
    def __init__(self, forceSize: tuple[int, int] | None = None, doDepth: bool = True, doPerson: bool = True) -> None:

        self.doDepth: bool = doDepth
        self.doPerson: bool = doPerson

        self.colorMutex: Lock = Lock()
        self.colorFrameNew: bool = False
        self.colorID: int = 0
        self.colorCallbacks: list = []
        self.depthMutex: Lock = Lock()
        self.depthFrameNew: bool = False
        self.depthId: int = 0
        self.depthCallbacks: list = []

        self.outWidth: int = 1920
        self.outHeight: int = 1080

        if (forceSize):
            self.outWidth, self.outHeight = forceSize
        self.RAW_outputColor: np.ndarray = np.zeros((self.outWidth, self.outHeight, 3), dtype=np.uint8)
        self.outputColor: np.ndarray = np.zeros((self.outWidth, self.outHeight, 3), dtype=np.uint8)
        self.outputDepth: np.ndarray = np.zeros((self.outWidth, self.outHeight), dtype=np.uint8)

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

        # TRANSFORMATION SETTINGS
        self.flipV: bool            = False
        self.flipH: bool            = False
        self.rotate90: bool         = False
        self.zoom: float            = 1.0
        self.offset: np.ndarray     = np.array([0.0, 0.0], dtype=np.float32)
        self.perspective: np.ndarray= np.array([0.0, 0.0], dtype=np.float32)

        self.device: dai.Device

        self.qColorImage:           dai.DataOutputQueue
        self.qColorControl:         dai.DataInputQueue
        self.qColorConfig:          dai.DataInputQueue

        self.qDepthImage:           dai.DataOutputQueue
        self.qDepthConfig:          dai.DataInputQueue
        self.qDepthTransform:       dai.DataInputQueue

        self.deviceOpen: bool = False
        self.capturing:  bool = False

    def __exit__(self) -> None:
        self.close()

    def open(self) -> bool:
        if self.deviceOpen: return True

        pipeline = dai.Pipeline()
        setupColor(pipeline, self.outWidth, self.outHeight)
        if self.doDepth: setupDepth(pipeline, self.outWidth, self.outHeight)
        try: self.device = dai.Device(pipeline)
        except Exception as e:
            print('could not open camera, error', e, 'try again')
            try: self.device = dai.Device(pipeline)
            except Exception as e:
                print('still could not open camera, error', e)
                return False

        self.qColorImage    = self.device.getOutputQueue(name='color_image', maxSize=4, blocking=False)
        self.qColorControl  = self.device.getInputQueue('color_control')
        self.qColorConfig   = self.device.getInputQueue('color_manip_config')

        if self.doDepth:
            self.qDepthImage    = self.device.getOutputQueue(name='disparity', maxSize=4, blocking=False)
            self.qDepthConfig   = self.device.getInputQueue('depth_config')
            self.qDepthTransform= self.device.getInputQueue('disparity_manip_config')

        self.deviceOpen = True
        return True

    def close(self) -> None:
        if not self.deviceOpen: return
        if self.capturing: self.stopCapture()
        self.deviceOpen = False

        self.device.close()
        self.qColorConfig.close()
        self.qColorControl.close()
        self.qColorImage.close()
        if self.doDepth:
            self.qDepthImage.close()
            self.qDepthConfig.close()
            self.qDepthTransform.close()

    def startCapture(self) -> None:
        if not self.deviceOpen:
            print('CamDepthAi:start', 'device is not open')
            return
        if self.capturing: return
        self.updateWarp
        self.colorId = self.qColorImage.addCallback(self.updateColor) #type:ignore

        if self.doDepth:
            self.depthId = self.qDepthImage.addCallback(self.updateDepth) #type:ignore

    def stopCapture(self) -> None:
        if not self.capturing: return
        self.qColorImage.removeCallback(self.colorId) #type:ignore

        if self.doDepth:
            self.qDepthImage.removeCallback(self.depthId) #type:ignore

    def updateColor(self, inColor) -> None:
        cv_raw_frame: np.ndarray = inColor.getCvFrame() #type:ignore
        cv_frame: np.ndarray = fit(cv_raw_frame, self.outWidth, self.outHeight)

        for c in self.colorCallbacks:
            c(cv_frame)
        with self.colorMutex:
            self.colorFrameNew = True
            self.outputColor = cv_frame
            self.RAW_outputColor = cv_raw_frame
        self.updateControlValues(inColor)

    def updateDepth(self, inDepth) -> None:
        if not self.doDepth: return
        depthFrame = inDepth.getFrame() #type:ignore
        cvFrame = (depthFrame * (255 / 95)).astype(np.uint8)
        for c in self.depthCallbacks: c(cvFrame)
        with self.depthMutex:
            self.depthFrameNew = True
            self.outputDepth = cvFrame

    def isColorFrameNew(self) -> bool:
        value: bool = self.colorFrameNew
        self.colorFrameNew = False
        return value

    def isDepthFrameNew(self) -> bool:
        value: bool = self.depthFrameNew
        self.depthFrameNew = False
        return value

    def getColorImgRaw(self) -> np.ndarray:
        with self.colorMutex:
            return self.RAW_outputColor

    def getColorImg(self) -> np.ndarray:
        with self.colorMutex:
            return self.outputColor

    def getDepthImg(self) -> np.ndarray:
        with self.depthMutex:
            return self.outputDepth

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


    def setAutoExposure(self, value) -> None:
        if not self.deviceOpen: return
        self.autoExposure = value
        if value == False:
            self.setExposureIso(self.exposure, self.iso)
            return
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureEnable()
        self.qColorControl.send(ctrl)

    def setAutoFocus(self, value) -> None:
        if not self.deviceOpen: return
        self.autoFocus = value
        if value == False:
            self.setFocus(self.focus)
            return
        ctrl = dai.CameraControl()
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
        self.qColorControl.send(ctrl)

    def setAutoWhiteBalance(self, value) -> None:
        if not self.deviceOpen: return
        self.autoWhiteBalance = value
        if value == False:
            self.setWhiteBalance(self.whiteBalance)
            return
        ctrl = dai.CameraControl()
        ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
        self.qColorControl.send(ctrl)

    def setExposureIso(self, exposure: int, iso: int) -> None:
        if not self.deviceOpen: return
        self.autoExposure = False
        self.exposure = int(clamp(exposure, exposureRange))
        self.iso = int(clamp(iso, isoRange))
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(int(self.exposure), int(self.iso))
        self.qColorControl.send(ctrl)

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
        self.qColorControl.send(ctrl)

    def setWhiteBalance(self, value: int) -> None:
        if not self.deviceOpen: return
        self.autoWhiteBalance = False
        ctrl = dai.CameraControl()
        self.whiteBalance = int(clamp(value, whiteBalanceRange))
        ctrl.setManualWhiteBalance(int(self.whiteBalance))
        self.qColorControl.send(ctrl)


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

        self.qDepthConfig.send(cnfg)

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
        if not self.deviceOpen: return
        offset: tuple[float, float] = (self.offset[0], self.offset[1])
        perspective: tuple[float, float] = (self.perspective[0], self.perspective[1])
        dW: list[dai.Point2f] = makeWarpList(self.flipH, self.flipV, self.rotate90, self.zoom, offset, perspective)
        config = dai.ImageManipConfig()
        config.setWarpTransformFourPoints(dW, True)
        # config.setResize(self.outWidth, self.outHeight)
        self.qColorConfig.send(config)
        if self.doDepth:
            self.qDepthTransform.send(config)

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

    def setPerspectiveH(self, perspectiveV: float) -> None:
        self.perspective[1] = perspectiveV
        self.updateWarp()

    def setPerspectiveV(self, perspectiveV: float) -> None:
        self.perspective[0] = perspectiveV
        self.updateWarp()

    def setOffsetH(self, offsetV: float) -> None:
        self.offset[0] = offsetV
        self.updateWarp()

    def setOffsetV(self, offsetV: float) -> None:
        self.offset[1] = offsetV
        self.updateWarp()

    def setColorCallback(self, callback) -> None:
        self.colorCallbacks.append(callback)

    def clearColorCallbacks(self) -> None:
        self.colorCallbacks = []

    def setDepthCallback(self, callback) -> None:
        self.depthCallbacks.append(callback)

















