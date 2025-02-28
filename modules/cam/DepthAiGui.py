from cv2 import ROTATE_180, ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE

from modules.cam.DepthAi import DepthAi as Cam
from modules.cam.DepthAi import PreviewType, PreviewTypeNames
from modules.cam.DepthAi import exposureRange, isoRange, whiteBalanceRange
from modules.cam.DepthAi import stereoDepthRange, stereoBrightnessRange
from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame


class DepthAiGui(Cam):
    def __init__(self, gui: Gui | None, doMono: bool = True) -> None:
        self.gui: Gui | None = gui
        super().__init__(doMono)

        elem: list = []
        elem.append([E(eT.CMBO, 'Preview',      super().setPreview,             PreviewTypeNames[0],        PreviewTypeNames, expand=False),
                     E(eT.CHCK, 'FlipH',         super().setFlipH,              False),
                     E(eT.CHCK, 'FlipV',         super().setFlipV,              False),
                     E(eT.CHCK, 'Balance ',     super().setAutoWhiteBalance,    True),
                     E(eT.SLDR, 'Bal',          super().setWhiteBalance,        0, whiteBalanceRange,       200)])
        elem.append([E(eT.CHCK, 'Exposure',     super().setAutoExposure,        False),
                     E(eT.SLDR, 'Exp',          super().setExposure,            0,     exposureRange,       500),
                     E(eT.TEXT, 'Iso'),
                     E(eT.SLDR, 'Iso',          super().setIso,                 0,          isoRange,       50)])
        elem.append([E(eT.TEXT, 'Stereo  Min'),
                     E(eT.SLDR, 'StereoMin',    super().setDepthTresholdMin,    stereoDepthRange[0],        stereoDepthRange,     50),
                     E(eT.TEXT, 'Max'),
                     E(eT.SLDR, 'StereoMax',    super().setDepthTresholdMax,    stereoDepthRange[1],        stereoDepthRange,     50)])
        elem.append([E(eT.TEXT, 'Stereo Brgs'),
                     E(eT.SLDR, 'B_Filter',     super().setStereoMinBrightness,  stereoBrightnessRange[0],   stereoBrightnessRange, 1)])

        elem.append([E(eT.TEXT, 'IR     Grid'),
                     E(eT.SLDR, 'LightGrid',    self.setIrGridLight,           0, [0,1], 0.05),
                     E(eT.TEXT, 'Fld'),
                     E(eT.SLDR, 'LightFlood',   self.setIrFloodLight,          0, [0,1], 0.05)])

        self.color_frame = Frame('CAMERA', elem, 200)

        elem: list = []
        self.stereo_frame = Frame('STEREO', elem, 100)


        self.prevautoExposure: bool      = self.autoExposure
        self.prevExposure: int           = self.exposure
        self.prevIso: int                = self.iso
        self.prevAutoWhiteBalance: bool  = self.autoWhiteBalance
        self.prevWhiteBalance: int       = self.whiteBalance


    def updateControlValues(self, frame) -> None:
        super().updateControlValues(frame)
        if not self.gui or not self.gui.isRunning(): return
        if (self.prevautoExposure != self.autoExposure) :
            self.prevautoExposure = self.autoExposure
            self.gui.updateElement('Exposure', self.autoExposure)

        if (self.prevAutoWhiteBalance != self.autoWhiteBalance) :
            self.prevAutoWhiteBalance = self.autoWhiteBalance
            self.gui.updateElement('Balance ', self.autoWhiteBalance)

        if self.autoExposure:
            if (self.prevExposure != self.exposure) :
                self.prevExposure = self.exposure
                self.gui.updateElement('Exp', self.exposure)
            if (self.prevIso != self.iso) :
                self.prevIso = self.iso
                self.gui.updateElement('Iso', self.iso)

        if self.autoWhiteBalance:
            if (self.prevWhiteBalance != self.whiteBalance) :
                self.prevWhiteBalance = self.whiteBalance
                self.gui.updateElement('Bal', self.whiteBalance)

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


    def get_gui_frame(self):
          return self.get_color_frame()

    def get_color_frame(self):
          return self.color_frame

    def get_stereo_frame(self):
          return self.stereo_frame