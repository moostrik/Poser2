import math

from modules.cam.DepthAi.CoreSettings import *
from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame

def closest_log_value(number: float) -> float:
    return 10 ** round(math.log10(number))

def getStepsFromRange(range: tuple[int, int]) -> float:
    roughSteps: float = float(range[1] - range[0]) / 100.0
    return closest_log_value(roughSteps)

def gsfr(range: tuple[int, int]) -> float:
    return getStepsFromRange(range)

class DepthAiGui(DepthAiSettings):
    def __init__(self, gui: Gui | None, modelPath:str, fps: int = 30, doColor: bool = True, doStereo: bool = True, doPerson: bool = True, lowres: bool = False, showStereo: bool = False) -> None:
        self.gui: Gui | None = gui
        super().__init__(modelPath, fps, doColor, doStereo, doPerson, lowres, showStereo)

        id: str = self.IDs
        elem: list = []
        elem.append([E(eT.TEXT, 'Exposure  '),
                     E(eT.SLDR, 'C_Exposure'+id,        super().setColorExposure,       exposureRange[0],       exposureRange,      gsfr(exposureRange)),
                     E(eT.TEXT, '       Iso'),
                     E(eT.SLDR, 'C_Iso'+id,             super().setColorIso,            isoRange[0],            isoRange,           gsfr(isoRange))])
        elem.append([E(eT.TEXT, 'Balance   '),
                     E(eT.SLDR, 'C_Balance'+id,         super().setColorBalance,        balanceRange[0],        balanceRange,       gsfr(balanceRange)),
                     E(eT.TEXT, '  Contrast'),
                     E(eT.SLDR, 'C_Contrast'+id,        super().setColorContrast,       contrastRange[0],       contrastRange,      gsfr(contrastRange))])
        elem.append([E(eT.TEXT, 'Brightness'),
                     E(eT.SLDR, 'C_Brightness'+id,      super().setColorBrightness,     brightnessRange[0],     brightnessRange,    gsfr(brightnessRange)),
                     E(eT.TEXT, '   Denoise'),
                     E(eT.SLDR, 'C_Denoise'+id,         super().setColorDenoise,        lumaDenoiseRange[0],    lumaDenoiseRange,   gsfr(lumaDenoiseRange))])
        elem.append([E(eT.TEXT, 'Saturation'),
                     E(eT.SLDR, 'C_Saturation'+id,      super().setColorSaturation,     saturationRange[0],     saturationRange,    gsfr(saturationRange)),
                     E(eT.TEXT, ' Sharpness'),
                     E(eT.SLDR, 'C_Sharpness'+id,       super().setColorSharpness,      sharpnessRange[0],      sharpnessRange,     gsfr(sharpnessRange))])
        elem.append([E(eT.CMBO, 'Preview',              super().setPreview,             self.getFrameNames()[0],  self.getFrameNames(),   expand=False),
                     E(eT.CHCK, 'AutoExposure'+id,      super().setColorAutoExposure,   True),
                     E(eT.CHCK, 'AutoBalance'+id,       super().setColorAutoBalance,    True),
                     E(eT.SLDR, 'FPS'+id,               None,                           0,    [0,60],  1)])
        elem.append([ E(eT.SLDR, 'NumTracklets'+id,      None,                           0,    [0,6],  1),
                     E(eT.SLDR, 'TPS'+id,               None,                           0,    [0,60],  1)])


        self.color_frame = Frame('CAMERA COLOR', elem, 240)

        elem: list = []
        elem.append([E(eT.TEXT, 'Exposure  '),
                     E(eT.SLDR, 'M_Exposure'+id,        super().setMonoExposure,        exposureRange[0],       exposureRange,      gsfr(exposureRange)),
                     E(eT.TEXT, 'Iso'),
                     E(eT.SLDR, 'M_Iso'+id,             super().setMonoIso,             isoRange[0],            isoRange,           gsfr(isoRange))])
        elem.append([E(eT.TEXT, 'Depth Min '),
                     E(eT.SLDR, 'S_Min'+id,             super().setDepthTresholdMin,    stereoDepthRange[0],    stereoDepthRange,   gsfr(stereoDepthRange)),
                     E(eT.TEXT, 'Max'),
                     E(eT.SLDR, 'S_Max'+id,             super().setDepthTresholdMax,    stereoDepthRange[0],    stereoDepthRange,   gsfr(stereoDepthRange))])
        elem.append([E(eT.TEXT, 'Bright Min'),
                     E(eT.SLDR, 'S_BrighnessMin'+id,    super().setStereoMinBrightness, stereoBrightnessRange[0], stereoBrightnessRange, gsfr(stereoBrightnessRange)),
                     E(eT.TEXT, 'Max'),
                     E(eT.SLDR, 'S_BrighnessMax'+id,    super().setStereoMaxBrightness, stereoBrightnessRange[0], stereoBrightnessRange, gsfr(stereoBrightnessRange))])
        elem.append([E(eT.TEXT, 'IR Grid   '),
                     E(eT.SLDR, 'L_Grid'+id,            self.setIrGridLight,            0, [0,1], 0.05),
                     E(eT.TEXT, 'Fld'),
                     E(eT.SLDR, 'L_Flood'+id,           self.setIrFloodLight,           0, [0,1], 0.05)])
        elem.append([E(eT.TEXT, 'Filter    '),
                     E(eT.CMBO, 'Median'+id,            super().setStereoMedianFilter,  StereoMedianFilterTypeNames[0],     StereoMedianFilterTypeNames, expand=False),
                     E(eT.TEXT, '              '),
                     E(eT.CHCK, 'M_AutoExposure'+id,    super().setMonoAutoExposure,    False)])

        self.depth_frame = Frame('CAMERA DEPTH', elem, 200)

        self.prevColorAutoExposure: bool =  self.colorAutoExposure
        self.prevColorExposure: int =       self.colorExposure
        self.prevColorIso: int =            self.colorIso
        self.prevColorAutoBalance: bool =   self.colorAutoBalance
        self.prevColorWhiteBalance: int =   self.colorBalance
        self.prevMonoAutoExposure: bool =   self.monoAutoExposure
        self.prevMonoExposure: int =        self.monoExposure
        self.prevMonoIso: int =             self.monoIso

    # COLOR
    def updateColorControl(self, frame) -> None: #override
        super().updateColorControl(frame)
        if not self.gui or not self.gui.isRunning(): return
        id: str = self.IDs

        if (self.prevColorAutoExposure != self.colorAutoExposure) :
            self.prevColorAutoExposure = self.colorAutoExposure
            self.gui.updateElement('AutoExposure'+id, self.colorAutoExposure)

        if (self.prevColorAutoBalance != self.colorAutoBalance) :
            self.prevColorAutoBalance = self.colorAutoBalance
            self.gui.updateElement('AutoBalance'+id, self.colorAutoBalance)

        if self.colorAutoExposure:
            if (self.prevColorExposure != self.colorExposure) :
                self.prevColorExposure = self.colorExposure
                self.gui.updateElement('C_Exposure'+id, self.colorExposure)
            if (self.prevColorIso != self.colorIso) :
                self.prevColorIso = self.colorIso
                self.gui.updateElement('C_Iso'+id, self.colorIso)

        if self.colorAutoBalance:
            if (self.prevColorWhiteBalance != self.colorBalance) :
                self.prevColorWhiteBalance = self.colorBalance
                self.gui.updateElement('C_Balance'+id, self.colorBalance)


    # MONO
    def updateMonoControl(self, frame) -> None: #override
        super().updateMonoControl(frame)
        if not self.gui or not self.gui.isRunning(): return
        id: str = self.IDs

        if (self.prevMonoAutoExposure != self.monoAutoExposure) :
            self.prevMonoAutoExposure = self.monoAutoExposure
            self.gui.updateElement('M_AutoExposure'+id, self.monoAutoExposure)

        if self.monoAutoExposure:
            if (self.prevMonoExposure != self.monoExposure) :
                self.prevMonoExposure = self.monoExposure
                self.gui.updateElement('M_Exposure'+id, self.monoExposure)
            if (self.prevMonoIso != self.monoIso) :
                self.prevMonoIso = self.monoIso
                self.gui.updateElement('M_Iso'+id, self.monoIso)

    # IR
    def setIrGridLight(self, value: float) -> None: #override
        super().setIrGridLight(value)
        if not self.gui or not self.gui.isRunning(): return
        if value > 0.0:
            self.gui.updateElement('L_Flood'+self.IDs, 0.0)
            super().setIrFloodLight(0.0)

    def setIrFloodLight(self, value: float) -> None: #override
        super().setIrFloodLight(value)
        if not self.gui or not self.gui.isRunning(): return
        if value > 0.0:
            self.gui.updateElement('L_Grid'+self.IDs, 0.0)
            super().setIrGridLight(0.0)

    # FPS
    def updateFPS(self) -> None: #override
        super().updateFPS()
        if not self.gui or not self.gui.isRunning(): return
        self.gui.updateElement('FPS'+self.IDs, self.getFPS())
        self.gui.updateElement('TPS'+self.IDs, self.getTPS())

        # in a perfect world, this would have its own defintion
        self.gui.updateElement('NumTracklets'+self.IDs,  self.numTracklets)

    # GUI FRAME
    def get_gui_frame(self):
          return self.get_gui_color_frame()

    def get_gui_color_frame(self):
          return self.color_frame

    def get_gui_depth_frame(self):
          return self.depth_frame