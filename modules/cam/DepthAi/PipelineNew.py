
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
    showMono: bool = False
    ) -> dai.RawStereoDepthConfig:

    stereoConfig: dai.RawStereoDepthConfig = dai.RawStereoDepthConfig()

    # SetupPipelineOld(pipeline, modelPath, fps, doColor, doStereo, doPerson, lowres, showLeft)
    # stereoConfig.algorithmControl.depthAlign = dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.CENTER
    # return stereoConfig

    # SetupColor(pipeline, fps, lowres)
    # return stereoConfig

    # SetupColorPerson(pipeline, fps, lowres, modelPath)
    # return stereoConfig

    # SetupColorStereo(pipeline, fps, lowres, showMono)
    # stereoConfig.algorithmControl.depthAlign = dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.CENTER
    # return stereoConfig

    # SetupColorStereoPerson(pipeline, fps, lowres, showMono, modelPath)
    # stereoConfig.algorithmControl.depthAlign = dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.CENTER
    # return stereoConfig


    if doColor:
        stereoConfig.algorithmControl.depthAlign = dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.CENTER
        if doStereo:
            if doPerson:
                SetupColorStereoPerson(pipeline, fps, lowres, showMono, modelPath)

            else:
                SetupColorStereo(pipeline, fps, lowres, showMono)
        else:
            if doPerson:
                SetupColorPerson(pipeline, fps, lowres, modelPath)
            else:
                SetupColor(pipeline, fps, lowres)
    else:
        stereoConfig.algorithmControl.depthAlign = dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT
        if doStereo:
            if doPerson:
                SetupMonoStereoPerson(pipeline, fps, lowres, showMono, modelPath)
            else:
                SetupMonoStereo(pipeline, fps, lowres, showMono)
        else:
            if doPerson:
                SetupMonoPerson(pipeline, fps, lowres, modelPath)
            else:
                SetupMono(pipeline, fps, lowres)

    return stereoConfig

class Setup():
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool = False) -> None:
        self.pipeline: dai.Pipeline = pipeline
        self.fps: int = fps
        self.lowres: bool = lowres

        self.sync: dai.node.Sync = pipeline.create(dai.node.Sync)
        syncThreshold = int(1250 / fps)
        self.sync.setSyncAttempts(2)
        self.sync.setSyncThreshold(timedelta(milliseconds=syncThreshold))

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
        self.color: dai.node.Camera = pipeline.create(dai.node.Camera)
        self.color.setCamera("color")
        self.color.setSize(1280, 720)
        self.color.setFps(self.fps)
        self.color.setMeshSource(dai.CameraProperties.WarpMeshSource.NONE)
        self.color.video.link(self.sync.inputs["video"])

        self.colorControl.out.link(self.color.inputControl)

class SetupColorPerson(SetupColor):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool, model_path) -> None:
        super().__init__(pipeline, fps, lowres)

        self.manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setResize(300, 300)
        self.manip.initialConfig.setKeepAspectRatio(False)
        self.manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        self.color.video.link(self.manip.inputImage)

        self.detectionNetwork: dai.node.MobileNetDetectionNetwork = pipeline.create(dai.node.MobileNetDetectionNetwork)
        nnPathDefault: Path = (Path(model_path) / MODEL6S).resolve().absolute()
        self.detectionNetwork.setBlobPath(nnPathDefault)
        self.detectionNetwork.setConfidenceThreshold(DETECTIONTHRESHOLD)
        self.detectionNetwork.setNumInferenceThreads(2)
        self.detectionNetwork.input.setBlocking(False)
        self.manip.out.link(self.detectionNetwork.input)
        # self.detectionNetwork.out.link(self.sync.inputs["detection"])

        self.objectTracker: dai.node.ObjectTracker = pipeline.create(dai.node.ObjectTracker)
        self.objectTracker.setDetectionLabelsToTrack([15])  # track only person
        self.objectTracker.setTrackerType(TRACKERTYPE)
        self.objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        if self.lowres:
            self.detectionNetwork.passthrough.link(self.objectTracker.inputTrackerFrame)
        else:
            self.color.video.link(self.objectTracker.inputTrackerFrame)
        self.detectionNetwork.passthrough.link(self.objectTracker.inputDetectionFrame)
        self.detectionNetwork.out.link(self.objectTracker.inputDetections)
        # self.objectTracker.out.link(self.sync.inputs["tracklets"])

        trackerOut: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        trackerOut.setStreamName("tracklets")
        self.objectTracker.out.link(trackerOut.input)

class SetupColorStereo(SetupColor):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool, showMono:bool) -> None:
        super().__init__(pipeline, fps, lowres)
        self.showMono: bool = showMono

        self.color.setMeshSource(dai.CameraProperties.WarpMeshSource.CALIBRATION)

        resolution: dai.MonoCameraProperties.SensorResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
        if lowres:
            resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P

        self.left: dai.node.MonoCamera = pipeline.create(dai.node.MonoCamera)
        self.left.setCamera("left")
        self.left.setResolution(resolution)
        self.left.setFps(fps)
        if self.showMono:
            self.left.out.link(self.sync.inputs["left"])

        self.right: dai.node.MonoCamera = pipeline.create(dai.node.MonoCamera)
        self.right.setCamera("right")
        self.right.setResolution(resolution)
        self.right.setFps(fps)
        if self.showMono:
            self.right.out.link(self.sync.inputs["right"])

        self.stereo: dai.node.StereoDepth = pipeline.create(dai.node.StereoDepth)
        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        self.stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setExtendedDisparity(False)
        self.stereo.setSubpixel(False)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.CENTER)

        self.left.out.link(self.stereo.left)
        self.right.out.link(self.stereo.right)
        self.stereo.disparity.link(self.sync.inputs["stereo"])

        self.monoControl.out.link(self.left.inputControl)
        self.monoControl.out.link(self.right.inputControl)
        self.stereoControl.out.link(self.stereo.inputConfig)

class SetupColorStereoPerson(SetupColorStereo):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool, showMono: bool, model_path) -> None:
        # self.fps = 20
        super().__init__(pipeline, fps, lowres, showMono)

        self.manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setResize(300, 300)
        self.manip.initialConfig.setKeepAspectRatio(False)
        self.manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        self.color.video.link(self.manip.inputImage)

        self.detectionNetwork: dai.node.MobileNetSpatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        nnPathDefault: Path = (Path(model_path) / MODEL5S).resolve().absolute()
        self.detectionNetwork.setBlobPath(nnPathDefault)
        self.detectionNetwork.setConfidenceThreshold(DETECTIONTHRESHOLD)
        self.detectionNetwork.setNumInferenceThreads(2)
        self.detectionNetwork.setBoundingBoxScaleFactor(0.5)
        self.detectionNetwork.setDepthLowerThreshold(100)
        self.detectionNetwork.setDepthUpperThreshold(5000)
        self.detectionNetwork.input.setBlocking(False)
        self.manip.out.link(self.detectionNetwork.input)
        self.stereo.depth.link(self.detectionNetwork.inputDepth)

        self.objectTracker: dai.node.ObjectTracker = pipeline.create(dai.node.ObjectTracker)
        self.objectTracker.setDetectionLabelsToTrack([15])  # track only person
        self.objectTracker.setTrackerType(TRACKERTYPE)
        self.objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        self.detectionNetwork.passthrough.link(self.objectTracker.inputTrackerFrame)
        self.detectionNetwork.passthrough.link(self.objectTracker.inputDetectionFrame)
        self.detectionNetwork.out.link(self.objectTracker.inputDetections)

        trackerOut: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        trackerOut.setStreamName("tracklets")
        self.objectTracker.out.link(trackerOut.input)


class SetupMono(Setup):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool) -> None:
        super().__init__(pipeline, fps, lowres)

        self.left: dai.node.MonoCamera = pipeline.create(dai.node.MonoCamera)
        self.resolution: dai.MonoCameraProperties.SensorResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
        if self.lowres:
            self.resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P
        self.left.setCamera("left")
        self.left.setResolution(self.resolution)
        self.left.setFps(self.fps)

        self.left.out.link(self.sync.inputs["video"])

        self.monoControl.out.link(self.left.inputControl)

class SetupMonoPerson(SetupMono):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool, model_path:str) -> None:
        super().__init__(pipeline, fps, lowres)

        self.manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setResize(300, 300)
        self.manip.initialConfig.setKeepAspectRatio(False)
        self.manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        self.left.out.link(self.manip.inputImage)

        self.detectionNetwork: dai.node.MobileNetDetectionNetwork = pipeline.create(dai.node.MobileNetDetectionNetwork)
        nnPathDefault: Path = (Path(model_path) / MODEL6S).resolve().absolute()
        self.detectionNetwork.setBlobPath(nnPathDefault)
        self.detectionNetwork.setConfidenceThreshold(DETECTIONTHRESHOLD)
        self.detectionNetwork.setNumInferenceThreads(2)
        self.detectionNetwork.input.setBlocking(False)
        self.manip.out.link(self.detectionNetwork.input)

        self.objectTracker: dai.node.ObjectTracker = pipeline.create(dai.node.ObjectTracker)
        self.objectTracker.setDetectionLabelsToTrack([15])  # track only person
        self.objectTracker.setTrackerType(TRACKERTYPE)
        self.objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        if self.lowres:
            self.manip.out.link(self.objectTracker.inputTrackerFrame)
        else:
            maxFrameSize = 1280 * 720 * 3
            self.manip2: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
            self.manip2.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
            self.manip2.initialConfig.setResize(1280, 720)
            self.manip2.initialConfig.setKeepAspectRatio(False)
            self.manip2.setMaxOutputFrameSize(maxFrameSize)
            self.left.out.link(self.manip2.inputImage)
            self.manip2.out.link(self.objectTracker.inputTrackerFrame)


        self.detectionNetwork.passthrough.link(self.objectTracker.inputDetectionFrame)
        self.detectionNetwork.out.link(self.objectTracker.inputDetections)

        trackerOut: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        trackerOut.setStreamName("tracklets")
        self.objectTracker.out.link(trackerOut.input)

class SetupMonoStereo(Setup):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool, showMono: bool) -> None:
        super().__init__(pipeline, fps, lowres)
        self.showMono: bool = showMono

        self.left: dai.node.MonoCamera = pipeline.create(dai.node.MonoCamera)
        self.resolution: dai.MonoCameraProperties.SensorResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
        if self.lowres:
            self.resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P
        self.left.setCamera("left")
        self.left.setResolution(self.resolution)
        self.left.setFps(self.fps)
        if self.showMono:
            self.right.out.link(self.sync.inputs["left"])

        self.right: dai.node.MonoCamera = pipeline.create(dai.node.MonoCamera)
        self.right.setCamera("right")
        self.right.setResolution(self.resolution)
        self.right.setFps(fps)
        if self.showMono:
            self.right.out.link(self.sync.inputs["right"])

        self.stereo: dai.node.StereoDepth = pipeline.create(dai.node.StereoDepth)
        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        self.stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setExtendedDisparity(False)
        self.stereo.setSubpixel(False)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.LEFT)

        self.left.out.link(self.stereo.left)
        self.right.out.link(self.stereo.right)
        self.stereo.disparity.link(self.sync.inputs["stereo"])

        self.stereo.rectifiedLeft.link(self.sync.inputs["video"])

        self.monoControl.out.link(self.right.inputControl)
        self.stereoControl.out.link(self.stereo.inputConfig)

class SetupMonoStereoPerson(SetupMonoStereo):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool, showMono: bool, model_path:str) -> None:
        super().__init__(pipeline, fps, lowres, showMono)

        self.manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setResize(300, 300)
        self.manip.initialConfig.setKeepAspectRatio(False)
        self.manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        self.stereo.rectifiedLeft.link(self.manip.inputImage)

        self.detectionNetwork: dai.node.MobileNetSpatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        nnPathDefault: Path = (Path(model_path) / MODEL5S).resolve().absolute()
        self.detectionNetwork.setBlobPath(nnPathDefault)
        self.detectionNetwork.setConfidenceThreshold(DETECTIONTHRESHOLD)
        self.detectionNetwork.setNumInferenceThreads(2)
        self.detectionNetwork.setBoundingBoxScaleFactor(0.5)
        self.detectionNetwork.setDepthLowerThreshold(100)
        self.detectionNetwork.setDepthUpperThreshold(5000)
        self.detectionNetwork.input.setBlocking(False)
        self.manip.out.link(self.detectionNetwork.input)
        self.stereo.depth.link(self.detectionNetwork.inputDepth)

        self.objectTracker: dai.node.ObjectTracker = pipeline.create(dai.node.ObjectTracker)
        self.objectTracker.setDetectionLabelsToTrack([15])  # track only person
        self.objectTracker.setTrackerType(TRACKERTYPE)
        self.objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        self.detectionNetwork.passthrough.link(self.objectTracker.inputTrackerFrame)
        self.detectionNetwork.passthrough.link(self.objectTracker.inputDetectionFrame)
        self.detectionNetwork.out.link(self.objectTracker.inputDetections)

        trackerOut: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        trackerOut.setStreamName("tracklets")
        self.objectTracker.out.link(trackerOut.input)


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
        # if (showLeft):
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

    sync.out.link(outputImages.input)

    return stereoconfig