from __future__ import annotations
from modules.cam.depthcam.Core import *
from modules.cam.depthcam.Definitions import FrameType
from modules.cam.depthcam.Pipeline import get_stereo_config, get_frame_types

class Settings():
    def __init__(self, core: Core) -> None:
        self.core: Core = core

        # COLOR SETTINGS
        self.color_auto_exposure: bool= True
        self.color_auto_focus: bool =   True
        self.color_auto_balance: bool = True
        self.color_exposure: int =      0
        self.color_iso: int =           0
        self.color_focus: int =         0
        self.color_balance: int =       0
        self.color_contrast: int =      0
        self.color_brightness: int =    0
        self.color_luma_denoise: int =  0
        self.color_saturation: int =    0
        self.color_sharpness: int =     0

        # MONO SETTINGS
        self.mono_auto_exposure: bool = True
        self.mono_auto_focus: bool =    True
        self.mono_exposure: int =       0
        self.mono_iso: int =            0

        # STEREO SETTINGS
        self.stereo_config: dai.RawStereoDepthConfig = get_stereo_config(self.core.do_color)


    def update_color_control(self, frame: dai.ImgFrame) -> None:
        if (self.color_auto_exposure):
            self.color_exposure = int(frame.getExposureTime().total_seconds() * 1000000)
            self.color_iso = frame.getSensitivity()
        if (self.color_auto_focus):
            self.color_focus = frame.getLensPosition()
        if (self.color_auto_balance):
            self.color_balance = frame.getColorTemperature()

    def update_mono_control(self, frame: dai.ImgFrame) -> None:
        if (self.mono_auto_exposure):
            self.mono_exposure = int(frame.getExposureTime().total_seconds() * 1000000)
            self.mono_iso = frame.getSensitivity()


    # GENERAL SETTINGS
    def set_preview(self, value: FrameType | int | str) -> None:
        if isinstance(value, str) and value in FRAME_TYPE_NAMES:
            self.core.preview_type = FrameType(FRAME_TYPE_NAMES.index(value))
        else:
            self.core.preview_type = FrameType(value)

    def get_frame_types(self) -> list[FrameType]:
        return list(self.core.frame_types)

    def get_frame_type_names(self) -> list[str]:
        type_list: list[FrameType] = self.get_frame_types()
        # type_list.sort(key=lambda x: x.value)
        return [preview.name for preview in type_list]

    def get_id_string(self) -> str:
        return self.core.id_string

    def get_num_tracklets(self) -> int:
        return self.core.num_tracklets

    # COLOR SETTINGS
    def set_color_auto_exposure(self, value) -> None:
        if not self.core.device_open: return
        self.color_auto_exposure = value
        if value == False:
            self.set_color_exposure_iso(self.color_exposure, self.color_iso)
            return
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureEnable()
        self.core.color_control.send(ctrl)

    def set_color_auto_balance(self, value) -> None:
        if not self.core.device_open: return
        self.color_auto_balance = value
        if value == False:
            self.set_color_balance(self.color_balance)
            return
        ctrl = dai.CameraControl()
        ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
        self.core.color_control.send(ctrl)

    def set_color_exposure_iso(self, exposure: int, iso: int) -> None:
        if not self.core.device_open: return
        self.color_auto_exposure = False
        self.color_exposure = int(self.clamp(exposure, EXPOSURE_RANGE))
        self.color_iso = int(self.clamp(iso, ISO_RANGE))
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(self.color_exposure, self.color_iso)
        self.core.color_control.send(ctrl)

    def set_color_exposure(self, value : int) -> None:
        self.set_color_exposure_iso(value, self.color_iso)

    def set_color_iso(self, value: int) -> None:
        self.set_color_exposure_iso(self.color_exposure, value)

    def set_color_balance(self, value: int) -> None:
        if not self.core.device_open: return
        self.color_auto_balance = False
        ctrl = dai.CameraControl()
        self.color_balance = int(self.clamp(value, BALANCE_RANGE))
        ctrl.setManualWhiteBalance(self.color_balance)
        self.core.color_control.send(ctrl)

    def set_color_contrast(self, value: int) -> None:
        if not self.core.device_open: return
        ctrl = dai.CameraControl()
        self.color_contrast = int(self.clamp(value, CONTRAST_RANGE))
        ctrl.setContrast(self.color_contrast)
        self.core.color_control.send(ctrl)

    def set_color_brightness(self, value: int) -> None:
        if not self.core.device_open: return
        ctrl = dai.CameraControl()
        self.color_brightness = int(self.clamp(value, BRIGHTNESS_RANGE))
        ctrl.setBrightness(self.color_brightness)
        self.core.color_control.send(ctrl)

    def set_color_denoise(self, value: int) -> None:
        if not self.core.device_open: return
        ctrl = dai.CameraControl()
        self.color_luma_denoise = int(self.clamp(value, LUMA_DENOISE_RANGE))
        ctrl.setLumaDenoise(self.color_luma_denoise)
        self.core.color_control.send(ctrl)

    def set_color_saturation(self, value: int) -> None:
        if not self.core.device_open: return
        ctrl = dai.CameraControl()
        self.color_saturation = int(self.clamp(value, SATURATION_RANGE))
        ctrl.setSaturation(self.color_saturation)
        self.core.color_control.send(ctrl)

    def set_color_sharpness(self, value: int) -> None:
        if not self.core.device_open: return
        ctrl = dai.CameraControl()
        self.color_sharpness = int(self.clamp(value, SHARPNESS_RANGE))
        ctrl.setSharpness(self.color_sharpness)
        self.core.color_control.send(ctrl)

    # MONO SETTINGS
    def set_mono_auto_exposure(self, value) -> None:
        if not self.core.device_open: return
        self.mono_auto_exposure = value
        if value == False:
            self.set_mono_exposure_iso(self.mono_exposure, self.mono_iso)
            return
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureEnable()
        self.core.mono_control.send(ctrl)

    def set_mono_exposure_iso(self, exposure: int, iso: int) -> None:
        if not self.core.device_open: return
        self.mono_auto_exposure = False
        self.mono_exposure = int(self.clamp(exposure, EXPOSURE_RANGE))
        self.mono_iso = int(self.clamp(iso, ISO_RANGE))
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(self.mono_exposure, self.mono_iso)
        self.core.mono_control.send(ctrl)

    def set_mono_exposure(self, value : int) -> None:
        self.set_mono_exposure_iso(value, self.mono_iso)

    def set_mono_iso(self, value: int) -> None:
        self.set_mono_exposure_iso(self.mono_exposure, value)

    # STEREO SETTINGS
    def set_depth_treshold_min(self, value: int) -> None:
        if not self.core.device_open: return
        v: int = int(self.clamp(value, STEREO_DEPTH_RANGE))
        self.stereo_config.postProcessing.thresholdFilter.minRange = int(v)
        self.core.stereo_control.send(self.stereo_config)

    def set_depth_treshold_max(self, value: int) -> None:
        if not self.core.device_open: return
        v: int = int(self.clamp(value, STEREO_DEPTH_RANGE))
        self.stereo_config.postProcessing.thresholdFilter.maxRange = int(v)
        self.core.stereo_control.send(self.stereo_config)

    def set_stereo_min_brightness(self, value: int) -> None:
        if not self.core.device_open: return
        v: int = int(self.clamp(value, STEREO_BRIGHTNESS_RANGE))
        self.stereo_config.postProcessing.brightnessFilter.minBrightness = int(v)
        self.core.stereo_control.send(self.stereo_config)

    def set_stereo_max_brightness(self, value: int) -> None:
        if not self.core.device_open: return
        v: int = int(self.clamp(value, STEREO_BRIGHTNESS_RANGE))
        self.stereo_config.postProcessing.brightnessFilter.maxBrightness = int(v)
        self.core.stereo_control.send(self.stereo_config)

    def set_stereo_median_filter(self, value: StereoMedianFilterType | int | str) -> None:
        if not self.core.device_open: return
        if isinstance(value, str):
            if value in STEREO_FILTER_NAMES:
                self.set_stereo_median_filter(StereoMedianFilterType(STEREO_FILTER_NAMES.index(value)))
            else:
                print('setStereoMedianFilter wrong type', value)
            return

        if value == StereoMedianFilterType.OFF:
            self.stereo_config.postProcessing.median = dai.MedianFilter.MEDIAN_OFF
        elif value == StereoMedianFilterType.KERNEL_3x3:
            self.stereo_config.postProcessing.median = dai.MedianFilter.KERNEL_3x3
        elif value == StereoMedianFilterType.KERNEL_5x5:
            self.stereo_config.postProcessing.median = dai.MedianFilter.KERNEL_5x5
        elif value == StereoMedianFilterType.KERNEL_7x7:
            self.stereo_config.postProcessing.median = dai.MedianFilter.KERNEL_7x7
        self.core.stereo_control.send(self.stereo_config)

    # IR SETTINGS
    def set_ir_flood_light(self, value: float) -> None:
        if not self.core.device_open: return
        v: float = self.clamp(value, (0.0, 1.0))
        self.core.device.setIrFloodLightIntensity(v)

    def set_ir_grid_light(self, value: float) -> None:
        if not self.core.device_open: return
        v: float = self.clamp(value, (0.0, 1.0))
        self.core.device.setIrLaserDotProjectorIntensity(v)

    # FPS
    def get_fps(self) -> float:
        return self.core.fps_counter.get_rate_average()

    def get_tps(self) -> float:
        return self.core.tps_counter.get_rate_average()

    # STATIC METHODS
    @staticmethod
    def clamp(num: int | float, size: tuple[int | float, int | float]) -> int | float:
        return max(size[0], min(num, size[1]))












