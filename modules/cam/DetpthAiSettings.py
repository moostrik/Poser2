from modules.cam.DepthAiCore import *

class DepthAiSettings(DepthAiCore):
    def __init__(self, modelPath:str, fps: int = 30, doColor: bool = True, doStereo: bool = True, doPerson: bool = True, lowres: bool = False, showLeft: bool = False) -> None:
        super().__init__(modelPath, fps, doColor, doStereo, doPerson, lowres, showLeft)


    # GENERAL SETTINGS
    def setPreview(self, value: PreviewType | int | str) -> None:
        if isinstance(value, str) and value in PreviewTypeNames:
            self.previewType = PreviewType(PreviewTypeNames.index(value))
        else:
            self.previewType = PreviewType(value)

    def setFlipH(self, flipH: bool) -> None:
        self.flipH = flipH

    def setFlipV(self, flipV: bool) -> None:
        self.flipV = flipV

    # COLOR SETTINGS
    def setColorAutoExposure(self, value) -> None:
        if not self.deviceOpen: return
        self.colorAutoExposure = value
        if value == False:
            self.setExposureIso(self.colorExposure, self.colorIso)
            return
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureEnable()
        self.colorControl.send(ctrl)

    def setColorAutoBalance(self, value) -> None:
        if not self.deviceOpen: return
        self.colorAutoBalance = value
        if value == False:
            self.setColorBalance(self.colorBalance)
            return
        ctrl = dai.CameraControl()
        ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
        self.colorControl.send(ctrl)

    def setExposureIso(self, exposure: int, iso: int) -> None:
        if not self.deviceOpen: return
        self.colorAutoExposure = False
        self.colorExposure = int(clamp(exposure, exposureRange))
        self.colorIso = int(clamp(iso, isoRange))
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(self.colorExposure, self.colorIso)
        self.colorControl.send(ctrl)

    def setColorExposure(self, value : int) -> None:
        self.setExposureIso(value, self.colorIso)

    def setColorIso(self, value: int) -> None:
        self.setExposureIso(self.colorExposure, value)

    def setColorBalance(self, value: int) -> None:
        if not self.deviceOpen: return
        self.colorAutoBalance = False
        ctrl = dai.CameraControl()
        self.colorBalance = int(clamp(value, balanceRange))
        ctrl.setManualWhiteBalance(self.colorBalance)
        self.colorControl.send(ctrl)

    def setColorContrast(self, value: int) -> None:
        if not self.deviceOpen: return
        ctrl = dai.CameraControl()
        self.colorContrast = int(clamp(value, contrastRange))
        ctrl.setContrast(self.colorContrast)
        self.colorControl.send(ctrl)

    def setColorBrightness(self, value: int) -> None:
        if not self.deviceOpen: return
        ctrl = dai.CameraControl()
        self.colorBrightness = int(clamp(value, brightnessRange))
        ctrl.setBrightness(self.colorBrightness)
        self.colorControl.send(ctrl)

    def setColorDenoise(self, value: int) -> None:
        if not self.deviceOpen: return
        ctrl = dai.CameraControl()
        self.colorLumaDenoise = int(clamp(value, lumaDenoiseRange))
        ctrl.setLumaDenoise(self.colorLumaDenoise)
        self.colorControl.send(ctrl)

    def setColorSaturation(self, value: int) -> None:
        if not self.deviceOpen: return
        ctrl = dai.CameraControl()
        self.colorSaturation = int(clamp(value, saturationRange))
        ctrl.setSaturation(self.colorSaturation)
        self.colorControl.send(ctrl)

    def setColorSharpness(self, value: int) -> None:
        if not self.deviceOpen: return
        ctrl = dai.CameraControl()
        self.colorSharpness = int(clamp(value, sharpnessRange))
        ctrl.setSharpness(self.colorSharpness)
        self.colorControl.send(ctrl)

    # MONO SETTINGS
    def setMonoAutoExposure(self, value) -> None:
        if not self.deviceOpen: return
        self.monoAutoExposure = value
        if value == False:
            self.setMonoExposureIso(self.monoExposure, self.monoIso)
            return
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureEnable()
        self.monoControl.send(ctrl)

    def setMonoExposureIso(self, exposure: int, iso: int) -> None:
        if not self.deviceOpen: return
        self.monoAutoExposure = False
        self.monoExposure = int(clamp(exposure, exposureRange))
        self.monoIso = int(clamp(iso, isoRange))
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(self.monoExposure, self.monoIso)
        self.monoControl.send(ctrl)

    def setMonoExposure(self, value : int) -> None:
        self.setMonoExposureIso(value, self.monoIso)

    def setMonoIso(self, value: int) -> None:
        self.setMonoExposureIso(self.monoExposure, value)

    # STEREO SETTINGS
    def setDepthTresholdMin(self, value: int) -> None:
        if not self.deviceOpen: return
        v: int = int(clamp(value, stereoDepthRange))
        self.stereoConfig.postProcessing.thresholdFilter.minRange = int(v)
        self.stereoControl.send(self.stereoConfig)

    def setDepthTresholdMax(self, value: int) -> None:
        if not self.deviceOpen: return
        v: int = int(clamp(value, stereoDepthRange))
        self.stereoConfig.postProcessing.thresholdFilter.maxRange = int(v)
        self.stereoControl.send(self.stereoConfig)

    def setStereoMinBrightness(self, value: int) -> None:
        if not self.deviceOpen: return
        v: int = int(clamp(value, stereoBrightnessRange))
        self.stereoConfig.postProcessing.brightnessFilter.minBrightness = int(v)
        self.stereoControl.send(self.stereoConfig)

    def setStereoMaxBrightness(self, value: int) -> None:
        if not self.deviceOpen: return
        v: int = int(clamp(value, stereoBrightnessRange))
        self.stereoConfig.postProcessing.brightnessFilter.maxBrightness = int(v)
        self.stereoControl.send(self.stereoConfig)

    def setStereoMedianFilter(self, value: StereoMedianFilterType | int | str) -> None:
        if not self.deviceOpen: return
        if isinstance(value, str):
            if value in StereoMedianFilterTypeNames:
                self.setStereoMedianFilter(StereoMedianFilterType(StereoMedianFilterTypeNames.index(value)))
            else:
                print('setStereoMedianFilter wrong type', value)
            return

        if value == StereoMedianFilterType.OFF:
            self.stereoConfig.postProcessing.median = dai.MedianFilter.MEDIAN_OFF
        elif value == StereoMedianFilterType.KERNEL_3x3:
            self.stereoConfig.postProcessing.median = dai.MedianFilter.KERNEL_3x3
        elif value == StereoMedianFilterType.KERNEL_5x5:
            self.stereoConfig.postProcessing.median = dai.MedianFilter.KERNEL_5x5
        elif value == StereoMedianFilterType.KERNEL_7x7:
            self.stereoConfig.postProcessing.median = dai.MedianFilter.KERNEL_7x7
        self.stereoControl.send(self.stereoConfig)

    # IR SETTINGS
    def setIrFloodLight(self, value: float) -> None:
        if not self.deviceOpen: return
        v: float = clamp(value, (0.0, 1.0))
        self.device.setIrFloodLightIntensity(v)

    def setIrGridLight(self, value: float) -> None:
        if not self.deviceOpen: return
        v: float = clamp(value, (0.0, 1.0))
        self.device.setIrLaserDotProjectorIntensity(v)













