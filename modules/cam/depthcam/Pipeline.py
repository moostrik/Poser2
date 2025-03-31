
import depthai as dai
from datetime import timedelta
from pathlib import Path

from modules.cam.depthcam.Definitions import *

def get_frame_types(do_color: bool, do_stereo: bool, show_stereo) -> list[FrameType]:
    frame_types: list[FrameType] = [FrameType.NONE]
    frame_types.append(FrameType.VIDEO)
    if do_stereo:
        frame_types.append(FrameType.LEFT)
        frame_types.append(FrameType.RIGHT)
        if show_stereo:
            frame_types.append(FrameType.STEREO)
    return frame_types

def get_stereo_config(do_color: bool) -> dai.RawStereoDepthConfig:
    stereoConfig: dai.RawStereoDepthConfig = dai.RawStereoDepthConfig()
    if do_color:
        stereoConfig.algorithmControl.depthAlign = dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.CENTER
    else:
        stereoConfig.algorithmControl.depthAlign = dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT
    return stereoConfig

def setup_pipeline(
    pipeline : dai.Pipeline,
    model_path:str,
    fps: int = 30,
    do_color: bool = True,
    do_stereo: bool = True,
    do_person: bool = True,
    lowres: bool = False,
    show_stereo: bool = False,
    simulate: bool = False
    ) -> None:

    options: list[str] = [
        'Color' if do_color else 'Mono',
        'Stereo' if do_stereo else '',
        'Yolo' if do_person else '',
        'LowRes' if lowres else 'Highres',
        'show_stereo' if show_stereo else '',
        'Simulate' if simulate else ''
    ]

    pipeline_description = "Depth Pipeline: " + " ".join(filter(None, options))
    print(pipeline_description)

    if not simulate:
        if do_color:
            if do_stereo:
                if do_person:
                    SetupColorStereoPerson(pipeline, fps, lowres, show_stereo, model_path)
                else:
                    SetupColorStereo(pipeline, fps, lowres, show_stereo)
            else:
                if do_person:
                    SetupColorPerson(pipeline, fps, lowres, model_path)
                else:
                    SetupColor(pipeline, fps, lowres)
        else:
            if do_stereo:
                if do_person:
                    SetupMonoStereoPerson(pipeline, fps, lowres, show_stereo, model_path)
                else:
                    SetupMonoStereo(pipeline, fps, lowres, show_stereo)
            else:
                if do_person:
                    SetupMonoPerson(pipeline, fps, lowres, model_path)
                else:
                    SetupMono(pipeline, fps, lowres)
    else:
        if do_color:
            if do_stereo:
                if do_person:
                    SimulationColorStereoPerson(pipeline, fps, lowres, show_stereo, model_path)
                else:
                    SimulationColorStereo(pipeline, fps, lowres, show_stereo)
            else:
                if do_person:
                    SimulationColorPerson(pipeline, fps, lowres, model_path)
                else:
                    SimulationColor(pipeline, fps, lowres)
        else:
            if do_stereo:
                if do_person:
                    SimulationMonoStereoPerson(pipeline, fps, lowres, show_stereo, model_path)
                else:
                    SimulationMonoStereo(pipeline, fps, lowres, show_stereo)
            else:
                if do_person:
                    SimulationMonoPerson(pipeline, fps, lowres, model_path)
                else:
                    SimulationMono(pipeline, fps, lowres)



class Setup():
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool = False) -> None:
        self.pipeline: dai.Pipeline = pipeline
        self.fps: int = fps
        self.lowres: bool = lowres


class SetupColor(Setup):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool) -> None:
        super().__init__(pipeline, fps, lowres)
        self.color: dai.node.Camera = pipeline.create(dai.node.Camera)
        self.color.setCamera("color")
        self.color.setSize(1280, 720)
        self.color.setFps(self.fps)
        self.color.setMeshSource(dai.CameraProperties.WarpMeshSource.NONE)

        self.output_video: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_video.setStreamName("video")
        self.color.video.link(self.output_video.input)

        self.color_control: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.color_control.setStreamName('color_control')
        self.color_control.out.link(self.color.inputControl)

class SetupColorPerson(SetupColor):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool, model_path: str) -> None:
        super().__init__(pipeline, fps, lowres)

        self.manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setResize(300, 300)
        self.manip.initialConfig.setKeepAspectRatio(False)
        self.manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        self.color.video.link(self.manip.inputImage)

        self.detection_network: dai.node.MobileNetDetectionNetwork = pipeline.create(dai.node.MobileNetDetectionNetwork)
        nn_path: Path = (Path(model_path) / DETECTION_MODEL6S).resolve().absolute()
        self.detection_network.setBlobPath(nn_path)
        self.detection_network.setConfidenceThreshold(DETECTION_THRESHOLD)
        self.detection_network.setNumInferenceThreads(2)
        self.detection_network.input.setBlocking(False)
        self.manip.out.link(self.detection_network.input)

        self.objectTracker: dai.node.ObjectTracker = pipeline.create(dai.node.ObjectTracker)
        self.objectTracker.setDetectionLabelsToTrack([15])  # track only person
        self.objectTracker.setTrackerType(TRACKER_TYPE)
        self.objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        if self.lowres:
            self.detection_network.passthrough.link(self.objectTracker.inputTrackerFrame)
        else:
            self.color.video.link(self.objectTracker.inputTrackerFrame)
        self.detection_network.passthrough.link(self.objectTracker.inputDetectionFrame)
        self.detection_network.out.link(self.objectTracker.inputDetections)

        self.outputTracklets: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.outputTracklets.setStreamName("tracklets")
        self.objectTracker.out.link(self.outputTracklets.input)

class SetupColorStereo(SetupColor):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool, show_stereo:bool) -> None:
        super().__init__(pipeline, fps, lowres)
        self.show_stereo: bool = show_stereo

        pipeline.remove(self.output_video)

        self.color.setMeshSource(dai.CameraProperties.WarpMeshSource.CALIBRATION)

        resolution: dai.MonoCameraProperties.SensorResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
        if lowres:
            resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P

        self.left: dai.node.MonoCamera = pipeline.create(dai.node.MonoCamera)
        self.left.setCamera("left")
        self.left.setResolution(resolution)
        self.left.setFps(fps)

        self.right: dai.node.MonoCamera = pipeline.create(dai.node.MonoCamera)
        self.right.setCamera("right")
        self.right.setResolution(resolution)
        self.right.setFps(fps)

        self.stereo: dai.node.StereoDepth = pipeline.create(dai.node.StereoDepth)
        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        self.stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setExtendedDisparity(False)
        self.stereo.setSubpixel(False)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.CENTER)
        self.left.out.link(self.stereo.left)
        self.right.out.link(self.stereo.right)

        self.sync: dai.node.Sync = pipeline.create(dai.node.Sync)
        sync_threshold = timedelta(seconds=(1.0 / self.fps) * 0.5)
        self.sync.setSyncThreshold(sync_threshold)

        self.color.video.link(self.sync.inputs["video"])
        self.left.out.link(self.sync.inputs["left"])
        self.right.out.link(self.sync.inputs["right"])
        if self.show_stereo:
            self.stereo.disparity.link(self.sync.inputs["stereo"])

        self.output_sync: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_sync.setStreamName("sync")
        self.sync.out.link(self.output_sync.input)

        self.mono_control: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.mono_control.setStreamName('mono_control')
        self.mono_control.out.link(self.left.inputControl)
        self.mono_control.out.link(self.right.inputControl)

        self.stereo_control: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.stereo_control.setStreamName('stereo_control')
        self.stereo_control.out.link(self.stereo.inputConfig)

class SetupColorStereoPerson(SetupColorStereo):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool, show_stereo: bool, model_path: str) -> None:
        super().__init__(pipeline, fps, lowres, show_stereo)

        self.manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setResize(300, 300)
        self.manip.initialConfig.setKeepAspectRatio(False)
        self.manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        self.color.video.link(self.manip.inputImage)

        self.detection_network: dai.node.MobileNetSpatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        nn_path: Path = (Path(model_path) / DETECTION_MODEL5S).resolve().absolute()
        self.detection_network.setBlobPath(nn_path)
        self.detection_network.setConfidenceThreshold(DETECTION_THRESHOLD)
        self.detection_network.setNumInferenceThreads(2)
        self.detection_network.setBoundingBoxScaleFactor(DEPTH_TRACKER_BOX_SCALE)
        self.detection_network.setDepthLowerThreshold(DEPTH_TRACKER_MIN_DEPTH)
        self.detection_network.setDepthUpperThreshold(DEPTH_TRACKER_MAX_DEPTH)
        self.detection_network.input.setBlocking(False)
        self.manip.out.link(self.detection_network.input)
        self.stereo.depth.link(self.detection_network.inputDepth)

        self.object_tracker: dai.node.ObjectTracker = pipeline.create(dai.node.ObjectTracker)
        self.object_tracker.setDetectionLabelsToTrack([15])  # track only person
        self.object_tracker.setTrackerType(TRACKER_TYPE)
        self.object_tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        self.detection_network.passthrough.link(self.object_tracker.inputTrackerFrame)
        self.detection_network.passthrough.link(self.object_tracker.inputDetectionFrame)
        self.detection_network.out.link(self.object_tracker.inputDetections)

        self.output_tracklets: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_tracklets.setStreamName("tracklets")
        self.object_tracker.out.link(self.output_tracklets.input)


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

        self.output_video: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_video.setStreamName("video")
        self.left.out.link(self.output_video.input)

        self.mono_control: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.mono_control.setStreamName('mono_control')
        self.mono_control.out.link(self.left.inputControl)

class SetupMonoPerson(SetupMono):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool, model_path: str) -> None:
        super().__init__(pipeline, fps, lowres)

        self.manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setResize(300, 300)
        self.manip.initialConfig.setKeepAspectRatio(False)
        self.manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        self.left.out.link(self.manip.inputImage)

        self.detection_network: dai.node.MobileNetDetectionNetwork = pipeline.create(dai.node.MobileNetDetectionNetwork)
        nn_path: Path = (Path(model_path) / DETECTION_MODEL6S).resolve().absolute()
        self.detection_network.setBlobPath(nn_path)
        self.detection_network.setConfidenceThreshold(DETECTION_THRESHOLD)
        self.detection_network.setNumInferenceThreads(2)
        self.detection_network.input.setBlocking(False)
        self.manip.out.link(self.detection_network.input)

        self.object_tracker: dai.node.ObjectTracker = pipeline.create(dai.node.ObjectTracker)
        self.object_tracker.setDetectionLabelsToTrack([15])  # track only person
        self.object_tracker.setTrackerType(TRACKER_TYPE)
        self.object_tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        if self.lowres:
            self.manip.out.link(self.object_tracker.inputTrackerFrame)
        else:
            max_frame_size = 1280 * 720 * 3
            self.manip2: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
            self.manip2.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
            self.manip2.initialConfig.setResize(1280, 720)
            self.manip2.initialConfig.setKeepAspectRatio(False)
            self.manip2.setMaxOutputFrameSize(max_frame_size)
            self.left.out.link(self.manip2.inputImage)
            self.manip2.out.link(self.object_tracker.inputTrackerFrame)

        self.detection_network.passthrough.link(self.object_tracker.inputDetectionFrame)
        self.detection_network.out.link(self.object_tracker.inputDetections)

        self.output_tracklets: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_tracklets.setStreamName("tracklets")
        self.object_tracker.out.link(self.output_tracklets.input)

class SetupMonoStereo(SetupMono):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool, show_stereo: bool) -> None:
        super().__init__(pipeline, fps, lowres)
        self.show_stereo: bool = show_stereo

        pipeline.remove(self.output_video)

        self.right: dai.node.MonoCamera = pipeline.create(dai.node.MonoCamera)
        self.right.setCamera("right")
        self.right.setResolution(self.resolution)
        self.right.setFps(fps)

        self.stereo: dai.node.StereoDepth = pipeline.create(dai.node.StereoDepth)
        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        self.stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setExtendedDisparity(False)
        self.stereo.setSubpixel(False)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.LEFT)

        self.left.out.link(self.stereo.left)
        self.right.out.link(self.stereo.right)

        self.sync: dai.node.Sync = pipeline.create(dai.node.Sync)
        sync_threshold = timedelta(seconds=(1.0 / self.fps) * 0.5)
        self.sync.setSyncThreshold(sync_threshold)

        self.stereo.rectifiedLeft.link(self.sync.inputs["video"])
        self.left.out.link(self.sync.inputs["left"])
        self.right.out.link(self.sync.inputs["right"])
        if self.show_stereo:
            self.stereo.disparity.link(self.sync.inputs["stereo"])

        self.output_sync: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_sync.setStreamName("sync")
        self.sync.out.link(self.output_sync.input)

        self.mono_control.out.link(self.right.inputControl)
        self.stereo_control: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.stereo_control.setStreamName('stereo_control')
        self.stereo_control.out.link(self.stereo.inputConfig)

class SetupMonoStereoPerson(SetupMonoStereo):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool, showMono: bool, model_path: str) -> None:
        super().__init__(pipeline, fps, lowres, showMono)

        self.manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setResize(300, 300)
        self.manip.initialConfig.setKeepAspectRatio(False)
        self.manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        self.stereo.rectifiedLeft.link(self.manip.inputImage)

        self.detection_network: dai.node.MobileNetSpatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        nn_path: Path = (Path(model_path) / DETECTION_MODEL5S).resolve().absolute()
        self.detection_network.setBlobPath(nn_path)
        self.detection_network.setConfidenceThreshold(DETECTION_THRESHOLD)
        self.detection_network.setNumInferenceThreads(2)
        self.detection_network.setBoundingBoxScaleFactor(DEPTH_TRACKER_BOX_SCALE)
        self.detection_network.setDepthLowerThreshold(DEPTH_TRACKER_MIN_DEPTH)
        self.detection_network.setDepthUpperThreshold(DEPTH_TRACKER_MAX_DEPTH)
        self.detection_network.input.setBlocking(False)
        self.manip.out.link(self.detection_network.input)
        self.stereo.depth.link(self.detection_network.inputDepth)

        self.object_tracker: dai.node.ObjectTracker = pipeline.create(dai.node.ObjectTracker)
        self.object_tracker.setDetectionLabelsToTrack([15])  # track only person
        self.object_tracker.setTrackerType(TRACKER_TYPE)
        self.object_tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        self.detection_network.passthrough.link(self.object_tracker.inputTrackerFrame)
        self.detection_network.passthrough.link(self.object_tracker.inputDetectionFrame)
        self.detection_network.out.link(self.object_tracker.inputDetections)

        self.output_tracklets: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_tracklets.setStreamName("tracklets")
        self.object_tracker.out.link(self.output_tracklets.input)


class SimulationColor(SetupColor):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool) -> None:
        super().__init__(pipeline, fps, lowres)

        pipeline.remove(self.color)

        self.ex_video: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_video.setStreamName("ex_video")
        self.ex_video.setMaxDataSize(1280*720*3)

        self.ex_video.out.link(self.output_video.input)

class SimulationColorPerson(SetupColorPerson):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool, model_path: str) -> None:
        super().__init__(pipeline, fps, lowres, model_path)

        pipeline.remove(self.color)

        self.ex_video: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_video.setStreamName("ex_video")
        self.ex_video.setMaxDataSize(1280*720*3)

        self.ex_video.out.link(self.manip.inputImage)
        if not self.lowres:
            self.color.video.link(self.objectTracker.inputTrackerFrame)

        self.ex_video.out.link(self.output_video.input)

class SimulationColorStereo(SetupColorStereo):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool, show_stereo: bool) -> None:
        super().__init__(pipeline, fps, lowres, show_stereo)

        pipeline.remove(self.left)
        pipeline.remove(self.right)
        pipeline.remove(self.sync)
        pipeline.remove(self.output_sync)
        pipeline.remove(self.output_video)
        pipeline.remove(self.color_control)
        pipeline.remove(self.mono_control)

        self.ex_video: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_video.setStreamName("ex_video")
        self.ex_video.setMaxDataSize(1280*720*3)

        self.ex_left: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_left.setStreamName("ex_left")
        self.ex_left.setMaxDataSize(1280*720*3)

        self.ex_right: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_right.setStreamName("ex_right")
        self.ex_right.setMaxDataSize(1280*720*3)

        self.ex_left.out.link(self.stereo.left)
        self.ex_right.out.link(self.stereo.right)

        self.output_video: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_video.setStreamName("video")
        self.ex_video.out.link(self.output_video.input)

        self.output_left: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_left.setStreamName("left")
        self.ex_left.out.link(self.output_left.input)

        self.output_right: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_right.setStreamName("right")
        self.ex_right.out.link(self.output_right.input)

        if self.show_stereo:
            self.output_stereo: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
            self.output_stereo.setStreamName("stereo")
            self.stereo.disparity.link(self.output_stereo.input)

class SimulationColorStereo_SYNC(SetupColorStereo):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool, show_stereo: bool) -> None:
        super().__init__(pipeline, fps, lowres, show_stereo)

        pipeline.remove(self.left)
        pipeline.remove(self.right)
        pipeline.remove(self.color_control)
        pipeline.remove(self.mono_control)

        self.ex_video: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_video.setStreamName("ex_video")
        self.ex_video.setMaxDataSize(1280*720*3)

        self.ex_left: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_left.setStreamName("ex_left")
        self.ex_left.setMaxDataSize(1280*720*3)

        self.ex_right: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_right.setStreamName("ex_right")
        self.ex_right.setMaxDataSize(1280*720*3)

        self.ex_left.out.link(self.stereo.left)
        self.ex_right.out.link(self.stereo.right)

        self.color.video.unlink(self.sync.inputs["color"])
        # self.ex_video.out.link(self.sync.inputs["color"])
        # self.ex_left.out.link(self.sync.inputs["left"])
        # self.ex_right.out.link(self.sync.inputs["right"])

class SimulationColorStereoPerson(SimulationColorStereo):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool, show_stereo: bool, model_path: str) -> None:
        super().__init__(pipeline, fps, lowres, show_stereo)

        self.manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setResize(300, 300)
        self.manip.initialConfig.setKeepAspectRatio(False)
        self.manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        self.ex_video.out.link(self.manip.inputImage)

        self.detection_network: dai.node.MobileNetSpatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        nn_path: Path = (Path(model_path) / DETECTION_MODEL5S).resolve().absolute()
        self.detection_network.setBlobPath(nn_path)
        self.detection_network.setConfidenceThreshold(DETECTION_THRESHOLD)
        self.detection_network.setNumInferenceThreads(2)
        self.detection_network.setBoundingBoxScaleFactor(DEPTH_TRACKER_BOX_SCALE)
        self.detection_network.setDepthLowerThreshold(DEPTH_TRACKER_MIN_DEPTH)
        self.detection_network.setDepthUpperThreshold(DEPTH_TRACKER_MAX_DEPTH)
        self.detection_network.input.setBlocking(False)
        self.manip.out.link(self.detection_network.input)
        self.stereo.depth.link(self.detection_network.inputDepth)

        self.objectTracker: dai.node.ObjectTracker = pipeline.create(dai.node.ObjectTracker)
        self.objectTracker.setDetectionLabelsToTrack([15])  # track only person
        self.objectTracker.setTrackerType(TRACKER_TYPE)
        self.objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        self.detection_network.passthrough.link(self.objectTracker.inputTrackerFrame)
        self.detection_network.passthrough.link(self.objectTracker.inputDetectionFrame)
        self.detection_network.out.link(self.objectTracker.inputDetections)

        self.trackerOut: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.trackerOut.setStreamName("tracklets")
        self.objectTracker.out.link(self.trackerOut.input)


class SimulationMono(SetupMono):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool) -> None:
        super().__init__(pipeline, fps, lowres)

        pipeline.remove(self.left)
        pipeline.remove(self.mono_control)

        self.ex_left: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_left.setStreamName("ex_video")
        self.ex_left.setMaxDataSize(1280*720*3)

        self.ex_left.out.link(self.output_video.input)

class SimulationMonoPerson(SetupMonoPerson):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool, model_path: str) -> None:
        super().__init__(pipeline, fps, lowres, model_path)

        pipeline.remove(self.left)
        pipeline.remove(self.mono_control)

        self.ex_left: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_left.setStreamName("ex_video")
        self.ex_left.setMaxDataSize(1280*720*3)

        self.ex_left.out.link(self.manip.inputImage)
        if not self.lowres:
            self.ex_left.out.link(self.object_tracker.inputTrackerFrame)

        self.ex_left.out.link(self.output_video.input)

class SimulationMonoStereo(SetupMonoStereo):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool, show_stereo: bool) -> None:
        super().__init__(pipeline, fps, lowres, show_stereo)

        pipeline.remove(self.left)
        pipeline.remove(self.right)
        pipeline.remove(self.sync)
        pipeline.remove(self.output_video)
        pipeline.remove(self.output_sync)
        pipeline.remove(self.mono_control)

        self.stereo.setDepthAlign(dai.CameraBoardSocket.CENTER)

        self.ex_video: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_video.setStreamName("ex_video")
        self.ex_video.setMaxDataSize(1280*720*3)

        self.ex_left: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_left.setStreamName("ex_left")
        self.ex_left.setMaxDataSize(1280*720*3)

        self.ex_right: dai.node.XLinkIn = pipeline.create(dai.node.XLinkIn)
        self.ex_right.setStreamName("ex_right")
        self.ex_right.setMaxDataSize(1280*720*3)

        self.ex_left.out.link(self.stereo.left)
        self.ex_right.out.link(self.stereo.right)

        self.output_video: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_video.setStreamName("video")
        self.ex_video.out.link(self.output_video.input)

        self.output_left: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_left.setStreamName("left")
        self.ex_left.out.link(self.output_left.input)

        self.output_right: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_right.setStreamName("right")
        self.ex_right.out.link(self.output_right.input)

        if self.show_stereo:
            self.output_stereo: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
            self.output_stereo.setStreamName("stereo")
            self.stereo.disparity.link(self.output_stereo.input)

class SimulationMonoStereoPerson(SimulationMonoStereo):
    def __init__(self, pipeline : dai.Pipeline, fps: int, lowres: bool, show_stereo: bool, model_path: str) -> None:
        super().__init__(pipeline, fps, lowres, show_stereo)

        self.color: dai.node.Camera = pipeline.create(dai.node.Camera)
        self.color.setCamera("color")
        self.color.setSize(1280, 720)
        self.color.setFps(self.fps)
        self.color.setMeshSource(dai.CameraProperties.WarpMeshSource.CALIBRATION)

        self.manip: dai.node.ImageManip = pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setResize(300, 300)
        self.manip.initialConfig.setKeepAspectRatio(False)
        self.manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        self.ex_video.out.link(self.manip.inputImage)

        self.detection_network: dai.node.MobileNetSpatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        nn_path: Path = (Path(model_path) / DETECTION_MODEL5S).resolve().absolute()
        self.detection_network.setBlobPath(nn_path)
        self.detection_network.setConfidenceThreshold(DETECTION_THRESHOLD)
        self.detection_network.setNumInferenceThreads(2)
        self.detection_network.setBoundingBoxScaleFactor(DEPTH_TRACKER_BOX_SCALE)
        self.detection_network.setDepthLowerThreshold(DEPTH_TRACKER_MIN_DEPTH)
        self.detection_network.setDepthUpperThreshold(DEPTH_TRACKER_MAX_DEPTH)
        self.detection_network.input.setBlocking(False)
        self.manip.out.link(self.detection_network.input)
        self.stereo.depth.link(self.detection_network.inputDepth)

        self.object_tracker: dai.node.ObjectTracker = pipeline.create(dai.node.ObjectTracker)
        self.object_tracker.setDetectionLabelsToTrack([15])  # track only person
        self.object_tracker.setTrackerType(TRACKER_TYPE)
        self.object_tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

        self.detection_network.passthrough.link(self.object_tracker.inputTrackerFrame)
        self.detection_network.passthrough.link(self.object_tracker.inputDetectionFrame)
        self.detection_network.out.link(self.object_tracker.inputDetections)

        self.output_tracklets: dai.node.XLinkOut = pipeline.create(dai.node.XLinkOut)
        self.output_tracklets.setStreamName("tracklets")
        self.object_tracker.out.link(self.output_tracklets.input)