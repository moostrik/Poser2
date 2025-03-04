from cv2 import ROTATE_180, ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE
import math

from modules.cam.DepthAi import DepthAi as Cam
from modules.cam.DepthAi import PreviewTypeNames, StereoMedianFilterTypeNames
from modules.cam.DepthAi import exposureRange, isoRange, whiteBalanceRange, contrastRange, brightnessRange, lumaDenoiseRange, saturationRange, sharpnessRange
from modules.cam.DepthAi import stereoDepthRange, stereoBrightnessRange
from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame

def closest_log_value(number: float) -> float:
    return 10 ** round(math.log10(number))

def getStepsFromRange(range: tuple[int, int]) -> float:
    roughSteps: float = float(range[1] - range[0]) / 100.0
    return closest_log_value(roughSteps)

def gsfr(range: tuple[int, int]) -> float:
    return getStepsFromRange(range)

class DepthAiGui(Cam):
    def __init__(self, gui: Gui | None, fps: int = 30, doMono: bool = True) -> None:
        self.gui: Gui | None = gui
        super().__init__(fps, doMono)

        elem: list = []
        elem.append([E(eT.TEXT, 'Exposure  '),
                     E(eT.SLDR, 'ColorExposure',        super().setColorExposure,       exposureRange[0],       exposureRange,      gsfr(exposureRange)),
                     E(eT.TEXT, '       Iso'),
                     E(eT.SLDR, 'ColorIso',             super().setColorIso,            isoRange[0],            isoRange,           gsfr(isoRange))])
        elem.append([E(eT.TEXT, 'Balance   '),
                     E(eT.SLDR, 'ColorBalance',         super().setColorBalance,        whiteBalanceRange[0],   whiteBalanceRange,  gsfr(whiteBalanceRange)),
                     E(eT.TEXT, '  Contrast'),
                     E(eT.SLDR, 'ColorContrast',        super().setColorContrast,       contrastRange[0],       contrastRange,      gsfr(contrastRange))])
        elem.append([E(eT.TEXT, 'Brightness'),
                     E(eT.SLDR, 'ColorBrightness',      super().setColorBrightness,     brightnessRange[0],     brightnessRange,    gsfr(brightnessRange)),
                     E(eT.TEXT, '   Denoise'),
                     E(eT.SLDR, 'ColorDenoise',         super().setColorDenoise,        lumaDenoiseRange[0],    lumaDenoiseRange,   gsfr(lumaDenoiseRange))])
        elem.append([E(eT.TEXT, 'Saturation'),
                     E(eT.SLDR, 'ColorSaturation',      super().setColorSaturation,     saturationRange[0],     saturationRange,    gsfr(saturationRange)),
                     E(eT.TEXT, ' Sharpness'),
                     E(eT.SLDR, 'ColorSharpness',       super().setColorSharpness,      sharpnessRange[0],      sharpnessRange,     gsfr(sharpnessRange))])
        elem.append([E(eT.CMBO, 'Preview',              super().setPreview,             PreviewTypeNames[0],    PreviewTypeNames,   expand=False),
                     E(eT.CHCK, ' FlipH',               super().setFlipH,               False),
                     E(eT.CHCK, ' FlipV',               super().setFlipV,               False),
                     E(eT.TEXT, ''),
                     E(eT.CHCK, 'AutoExposure',         super().setColorAutoExposure,   True),
                     E(eT.CHCK, 'AutoBalance',          super().setColorAutoBalance,    True)])

        self.color_frame = Frame('CAMERA COLOR', elem, 200)

        elem: list = []
        elem.append([E(eT.TEXT, 'Exposure  '),
                     E(eT.SLDR, 'MonoExposure',         super().setMonoExposure,        exposureRange[0],       exposureRange,      gsfr(exposureRange)),
                     E(eT.TEXT, 'Iso'),
                     E(eT.SLDR, 'MonoIso',              super().setMonoIso,             isoRange[0],            isoRange,           gsfr(isoRange))])
        elem.append([E(eT.TEXT, 'Depth Min '),
                     E(eT.SLDR, 'StereoMin',            super().setDepthTresholdMin,    stereoDepthRange[0],    stereoDepthRange,   gsfr(stereoDepthRange)),
                     E(eT.TEXT, 'Max'),
                     E(eT.SLDR, 'StereoMax',            super().setDepthTresholdMax,    stereoDepthRange[0],    stereoDepthRange,   gsfr(stereoDepthRange))])
        elem.append([E(eT.TEXT, 'Bright Min'),
                     E(eT.SLDR, 'StereoBrighnessMin',   super().setStereoMinBrightness, stereoBrightnessRange[0], stereoBrightnessRange, gsfr(stereoBrightnessRange)),
                     E(eT.TEXT, 'Max'),
                     E(eT.SLDR, 'StereoBrighnessMax',   super().setStereoMaxBrightness, stereoBrightnessRange[0], stereoBrightnessRange, gsfr(stereoBrightnessRange))])
        elem.append([E(eT.TEXT, 'IR Grid   '),
                     E(eT.SLDR, 'LightGrid',            self.setIrGridLight,            0, [0,1], 0.05),
                     E(eT.TEXT, 'Fld'),
                     E(eT.SLDR, 'LightFlood',           self.setIrFloodLight,           0, [0,1], 0.05)])
        elem.append([E(eT.TEXT, 'Filter    '),
                     E(eT.CMBO, 'Median',               super().setStereoMedianFilter,  StereoMedianFilterTypeNames[0],     StereoMedianFilterTypeNames, expand=False),
                     E(eT.TEXT, '    '),
                     E(eT.CHCK, 'MonoAutoExposure',     super().setMonoAutoExposure,    False)])

        self.depth_frame = Frame('CAMERA DEPTH', elem, 100)

        self.prevColorAutoExposure: bool =  self.colorAutoExposure
        self.prevColorExposure: int =       self.colorExposure
        self.prevColorIso: int =            self.colorIso
        self.prevColorAutoBalance: bool =   self.colorAutoBalance
        self.prevColorWhiteBalance: int =   self.colorWhiteBalance
        self.prevMonoAutoExposure: bool =   self.monoAutoExposure
        self.prevMonoExposure: int =        self.monoExposure
        self.prevMonoIso: int =             self.monoIso

    # COLOR
    def updateColorControl(self, frame) -> None: #override
        super().updateColorControl(frame)
        if not self.gui or not self.gui.isRunning(): return
        if (self.prevColorAutoExposure != self.colorAutoExposure) :
            self.prevColorAutoExposure = self.colorAutoExposure
            self.gui.updateElement('AutoExposure', self.colorAutoExposure)

        if (self.prevColorAutoBalance != self.colorAutoBalance) :
            self.prevColorAutoBalance = self.colorAutoBalance
            self.gui.updateElement('AutoBalance', self.colorAutoBalance)

        if self.colorAutoExposure:
            if (self.prevColorExposure != self.colorExposure) :
                self.prevColorExposure = self.colorExposure
                self.gui.updateElement('ColorExposure', self.colorExposure)
            if (self.prevColorIso != self.colorIso) :
                self.prevColorIso = self.colorIso
                self.gui.updateElement('ColorIso', self.colorIso)

        if self.colorAutoBalance:
            if (self.prevColorWhiteBalance != self.colorWhiteBalance) :
                self.prevColorWhiteBalance = self.colorWhiteBalance
                self.gui.updateElement('ColorBalance', self.colorWhiteBalance)

    # MONO
    def updateMonoControl(self, frame) -> None: #override
        super().updateMonoControl(frame)
        if not self.gui or not self.gui.isRunning(): return
        if (self.prevMonoAutoExposure != self.monoAutoExposure) :
            self.prevMonoAutoExposure = self.monoAutoExposure
            self.gui.updateElement('MonoAutoExposure', self.monoAutoExposure)

        if self.monoAutoExposure:
            if (self.prevMonoExposure != self.monoExposure) :
                self.prevMonoExposure = self.monoExposure
                self.gui.updateElement('MonoExposure', self.monoExposure)
            if (self.prevMonoIso != self.monoIso) :
                self.prevMonoIso = self.monoIso
                self.gui.updateElement('MonoIso', self.monoIso)

    # IR
    def setIrGridLight(self, value: float) -> None: #override
        super().setIrGridLight(value)
        if not self.gui or not self.gui.isRunning(): return
        if value > 0.0:
            self.gui.updateElement('LightFlood', 0.0)
            super().setIrFloodLight(0.0)

    def setIrFloodLight(self, value: float) -> None: #override
        super().setIrFloodLight(value)
        if not self.gui or not self.gui.isRunning(): return
        if value > 0.0:
            self.gui.updateElement('LightGrid', 0.0)
            super().setIrGridLight(0.0)

    # GUI FRAME
    def get_gui_frame(self):
          return self.get_gui_color_frame()

    def get_gui_color_frame(self):
          return self.color_frame

    def get_gui_depth_frame(self):
          return self.depth_frame
