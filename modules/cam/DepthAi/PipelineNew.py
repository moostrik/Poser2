
import depthai as dai
from datetime import timedelta
from pathlib import Path

MODEL5S: str = "mobilenet-ssd_openvino_2021.4_5shave.blob"
MODEL6S: str = "mobilenet-ssd_openvino_2021.4_6shave.blob"
DETECTIONTHRESHOLD: float = 0.5
TRACKERTYPE: dai.TrackerType = dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM
# ZERO_TERM_COLOR_HISTOGRAM higher accuracy (but can drift when losing object)
# ZERO_TERM_IMAGELESS slightly faster

def SetupPipeline(
    pipeline : dai.Pipeline,
    modelPath:str,
    fps: int = 30,
    doColor: bool = True,
    doStereo: bool = True,
    doPerson: bool = True,
    lowres: bool = False,
    showLeft: bool = False
    ) -> dai.RawStereoDepthConfig:

    stereoConfig: dai.RawStereoDepthConfig = dai.RawStereoDepthConfig()


    #SetupColor(pipeline, fps, lowres)
    #SetupColorPerson(pipeline, fps, lowres, modelPath)
    SetupColorStereo(pipeline, fps, lowres)

    return stereoConfig
    # setup = SetupColorStereo(pipeline, fps, lowres)
    # stereoConfig.algorithmControl.depthAlign = dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.CENTER
    # return stereoConfig
    # return setup.getStereoConfig()


    if doColor:
        if doStereo:
            if doPerson:
                stereoConfig = SetupColorStereoPerson(pipeline, fps, lowres, modelPath)
            else:
                stereoConfig = SetupColorStereo(pipeline, fps, lowres, modelPath)
        else:
            if doPerson:
                SetupColorPerson(pipeline, fps, lowres)
            else:
                SetupColor(pipeline, fps, lowres)
    else:
        if doStereo:
            if doPerson:
                stereoConfig = SetupMonoStereoPerson(pipeline, fps, lowres, modelPath)
            else:
                stereoConfig = SetupMonoStereo(pipeline, fps, lowres, modelPath)
        else:
            if doPerson:
                SetupMonoPerson(pipeline, fps, lowres)
            else:
                SetupMono(pipeline, fps, lowres)

    return stereoConfig

class Setup():
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool = False) -> None:
        self.pipeline: dai.Pipeline = pipeline
        self.fps: int = fps
        self.lowres: bool = lowres

        self.sync: dai.node.Sync = pipeline.create(dai.node.Sync)

        # CONTROL INPUTS
        self.colorControl: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.colorControl.setStreamName('color_control')

        self.monoControl: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.monoControl.setStreamName('mono_control')

        self.stereoControl: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.stereoControl.setStreamName('stereo_control')

        # OUTPUTS
        self.outputImages: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.outputImages.setStreamName("output_images")

        # LINKING
        self.sync.out.link(self.outputImages.input)

class SetupColor(Setup):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool) -> None:
        super().__init__(pipeline, fps, lowres)

        self.color: dai.node.ColorCamera = pipeline.create(dai.node.ColorCamera)
        self.color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
        self.color.setFps(fps)
        if lowres:
            self.color.setVideoSize(640, 360)

        self.color.video.link(self.sync.inputs["video"])
        self.sync.out.link(self.outputImages.input)

        self.colorControl.out.link(self.color.inputControl)

class SetupColorPerson(SetupColor):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool, model_path) -> None:
        super().__init__(pipeline, fps, lowres)

        self.manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setResize(300, 300)
        self.manip.initialConfig.setKeepAspectRatio(False)
        self.manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)

        self.detectionNetwork: dai.node.MobileNetDetectionNetwork = pipeline.create(dai.node.MobileNetDetectionNetwork)
        nnPathDefault: Path = (Path(model_path) / MODEL6S).resolve().absolute()
        self.detectionNetwork.setBlobPath(nnPathDefault)
        self.detectionNetwork.setConfidenceThreshold(DETECTIONTHRESHOLD)
        self.detectionNetwork.setNumInferenceThreads(2)
        self.detectionNetwork.input.setBlocking(True)

        self.objectTracker: dai.node.ObjectTracker = pipeline.create(dai.node.ObjectTracker)
        self.objectTracker.setDetectionLabelsToTrack([15])  # track only person
        self.objectTracker.setTrackerType(TRACKERTYPE)
        self.objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        self.color.video.link(self.manip.inputImage)
        self.manip.out.link(self.detectionNetwork.input)
        self.detectionNetwork.out.link(self.sync.inputs["detection"])

        self.detectionNetwork.passthrough.link(self.objectTracker.inputTrackerFrame)
        self.detectionNetwork.passthrough.link(self.objectTracker.inputDetectionFrame)
        self.detectionNetwork.out.link(self.objectTracker.inputDetections)
        self.objectTracker.out.link(self.sync.inputs["tracklets"])

class SetupColorStereo(Setup):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool) -> None:
        super().__init__(pipeline, fps, lowres)


        self.color: dai.node.Camera = pipeline.create(dai.node.Camera)
        self.left: dai.node.MonoCamera = pipeline.create(dai.node.MonoCamera)
        self.right: dai.node.MonoCamera = pipeline.create(dai.node.MonoCamera)

        self.color.setCamera("color")
        self.color.setSize(1280, 720)
        if lowres:
            self.color.setSize(640, 360)
        self.color.setFps(fps)
        self.color.setMeshSource(dai.CameraProperties.WarpMeshSource.CALIBRATION)
        # color.setBoardSocket(socket)
            # For now, RGB needs fixed focus to properly align with depth.
        # This value was used during calibration
        # try:
        #     calibData = readCalibration2()
        #     lensPosition = calibData.getLensPosition(RGB_SOCKET)
        #     if lensPosition:
        #         camRgb.initialControl.setManualFocus(lensPosition)
        # except:
        #     raise

        resolution: dai.MonoCameraProperties.SensorResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
        if lowres:
            resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P
        self.left.setCamera("left")
        self.left.setResolution(resolution)
        self.left.setFps(fps)

        self.right.setCamera("right")
        self.right.setResolution(resolution)
        self.right.setFps(fps)

        self.stereo: dai.node.StereoDepth = pipeline.create(dai.node.StereoDepth)
        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        self.stereo.setRectifyEdgeFillColor(0)  # black, to better see the cutout
        self.stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setExtendedDisparity(False)
        self.stereo.setSubpixel(False)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        # stereoConfig: dai.RawStereoDepthConfig = self.stereo.initialConfig.get()
        # stereoConfig.algorithmControl.depthAlign = dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.CENTER
        # self.stereo.initialConfig.set(stereoConfig)
        # self.stereoConfig: dai.RawStereoDepthConfig = stereoConfig

        self.left.out.link(self.stereo.left)
        self.right.out.link(self.stereo.right)

        self.color.video.link(self.sync.inputs["video"])
        self.stereo.disparity.link(self.sync.inputs["stereo"])
        # self.left.out.link(self.sync.inputs["left"])


        # self.colorControl.out.link(self.color.inputControl)
        # self.monoControl.out.link(self.left.inputControl)
        # self.monoControl.out.link(self.right.inputControl)
        # self.stereoControl.out.link(self.stereo.inputConfig)

    # def getStereoConfig(self) -> dai.RawStereoDepthConfig:
    #     return self.stereoConfig

def SetupColorStereoPerson (pipeline : dai.Pipeline, fps: int, lowres: bool, modelPath:str) -> dai.RawStereoDepthConfig:
    return dai.RawStereoDepthConfig()


def SetupMono (pipeline : dai.Pipeline, fps: int, lowres: bool) -> None:
    pass

def SetupMonoPerson (pipeline : dai.Pipeline, fps: int, lowres: bool) -> None:
    pass

def SetupMonoStereo (pipeline : dai.Pipeline, fps: int, lowres: bool, modelPath:str) -> dai.RawStereoDepthConfig:
    return dai.RawStereoDepthConfig()

def SetupMonoStereoPerson (pipeline : dai.Pipeline, fps: int, lowres: bool, modelPath:str) -> dai.RawStereoDepthConfig:
    return dai.RawStereoDepthConfig()


def SetupPipelineOld(
    pipeline : dai.Pipeline,
    modelPath:str,
    fps: int = 30,
    doColor: bool = True,
    doStereo: bool = True,
    doPerson: bool = True,
    lowres: bool = False,
    showLeft: bool = False
    ) -> dai.RawStereoDepthConfig:

    print('Depth Pipeline in', 'Color' if doColor else 'Mono', 'Stereo' if doStereo else '', 'Yolo' if doPerson else '', 'LowRes' if lowres else 'Highres', 'with ShowLeft' if showLeft else '')
    # MAIN NODES
    if doColor:
        color: dai.node.Camera = pipeline.create(dai.node.Camera)
    left: dai.node.MonoCamera = pipeline.create(dai.node.MonoCamera)

    if doStereo:
        right: dai.node.MonoCamera = pipeline.create(dai.node.MonoCamera)
        stereo: dai.node.StereoDepth = pipeline.create(dai.node.StereoDepth)
    sync: dai.node.Sync = pipeline.create(dai.node.Sync)

    if doPerson:
        manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        detectionNetwork: dai.node.MobileNetDetectionNetwork = pipeline.create(dai.node.MobileNetDetectionNetwork)
        objectTracker: dai.node.ObjectTracker = pipeline.create(dai.node.ObjectTracker)

    # CONTROL INPUTS
    colorControl: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
    colorControl.setStreamName('color_control')
    if doColor:
        colorControl.out.link(color.inputControl)

    monoControl: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
    monoControl.setStreamName('mono_control')
    monoControl.out.link(left.inputControl)
    if doStereo:
        monoControl.out.link(right.inputControl)

    stereoControl: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
    stereoControl.setStreamName('stereo_control')
    if doStereo:
        stereoControl.out.link(stereo.inputConfig)

    # OUTPUTS
    outputImages: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
    outputImages.setStreamName("output_images")

    # SETTINGS
    if doColor:
        color.setCamera("color")
        color.setSize(1280, 720)
        if lowres:
            color.setSize(640, 360)

        color.setFps(fps)
        if doStereo:
            color.setMeshSource(dai.CameraProperties.WarpMeshSource.CALIBRATION)
            # color.setBoardSocket(socket)
                # For now, RGB needs fixed focus to properly align with depth.
            # This value was used during calibration
            # try:
            #     calibData = readCalibration2()
            #     lensPosition = calibData.getLensPosition(RGB_SOCKET)
            #     if lensPosition:
            #         camRgb.initialControl.setManualFocus(lensPosition)
            # except:
            #     raise

    resolution: dai.MonoCameraProperties.SensorResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
    if lowres:
        resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P
    left.setCamera("left")
    left.setResolution(resolution)
    left.setFps(fps)
    if doStereo:
        right.setCamera("right")
        right.setResolution(resolution)
        right.setFps(fps)

    stereoconfig: dai.RawStereoDepthConfig = dai.RawStereoDepthConfig()
    if doStereo:
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setRectifyEdgeFillColor(0)  # black, to better see the cutout
        stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(False)

        if doColor:
            stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
            stereoconfig = stereo.initialConfig.get()
            stereoconfig.algorithmControl.depthAlign = dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.CENTER
        else:
            stereo.setDepthAlign(dai.CameraBoardSocket.LEFT)
            stereoconfig = stereo.initialConfig.get()
            stereoconfig.algorithmControl.depthAlign = dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT
        stereo.initialConfig.set(stereoconfig)

    syncThreshold = int(1250 / fps)
    sync.setSyncThreshold(timedelta(milliseconds=syncThreshold))

    if doPerson:
        manip.initialConfig.setResize(300, 300)
        manip.initialConfig.setKeepAspectRatio(False)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)

        nnPathDefault: Path = (Path(modelPath) / MODEL6S).resolve().absolute()
        if doStereo:
            nnPathDefault: Path = (Path(modelPath) / MODEL5S).resolve().absolute()
        detectionNetwork.setBlobPath(nnPathDefault)
        detectionNetwork.setConfidenceThreshold(DETECTIONTHRESHOLD)
        detectionNetwork.setNumInferenceThreads(2)
        detectionNetwork.input.setBlocking(False)

        objectTracker.setDetectionLabelsToTrack([15])  # track only person
        # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
        objectTracker.setTrackerType(TRACKERTYPE)
        objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

    # LINKING
    if doColor:
        color.video.link(sync.inputs["video"])
    else:
        if doStereo:
            stereo.rectifiedLeft.link(sync.inputs["video"])
        else:
            left.out.link(sync.inputs["video"])

    if doStereo:
        left.out.link(stereo.left)
        right.out.link(stereo.right)
        stereo.disparity.link(sync.inputs["stereo"])
        if (showLeft):
            left.out.link(sync.inputs["left"])

    if doPerson:
        if doColor:
            color.video.link(manip.inputImage)
        else:
            left.out.link(manip.inputImage)
        manip.out.link(detectionNetwork.input)
        detectionNetwork.out.link(sync.inputs["detection"])

        # objectTracker.passthroughTrackerFrame.link(manip.inputImage)
        detectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)
        detectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
        detectionNetwork.out.link(objectTracker.inputDetections)
        objectTracker.out.link(sync.inputs["tracklets"])

    color.preview.link(sync.inputs["left"])

    sync.out.link(outputImages.input)

    return stereoconfig