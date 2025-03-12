
import depthai as dai
from datetime import timedelta
from pathlib import Path

model5S: str = "mobilenet-ssd_openvino_2021.4_5shave.blob"
model6S: str = "mobilenet-ssd_openvino_2021.4_6shave.blob"
DetectionConfidenceThreshold: float = 0.5


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
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)

        nnPathDefault: Path = (Path(modelPath) / model6S).resolve().absolute()
        if doStereo:
            nnPathDefault: Path = (Path(modelPath) / model5S).resolve().absolute()
        detectionNetwork.setBlobPath(nnPathDefault)
        detectionNetwork.setConfidenceThreshold(DetectionConfidenceThreshold)
        detectionNetwork.setNumInferenceThreads(2)
        detectionNetwork.input.setBlocking(False)

        objectTracker.setDetectionLabelsToTrack([15])  # track only person
        objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
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

    sync.out.link(outputImages.input)

    return stereoconfig