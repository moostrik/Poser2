from cv2 import ROTATE_180, ROTATE_90_CLOCKWISE, ROTATE_90_COUNTERCLOCKWISE

from modules.cam.DepthAi import DepthAi as Cam
from modules.cam.DepthAi import PreviewType, PreviewTypeNames
from modules.cam.DepthAi import exposureRange, isoRange, focusRange, whiteBalanceRange
from modules.cam.DepthAi import depthDecimationRange, depthSpeckleRange, depthTresholdRange, depthHoleFillingRange, depthHoleIterRange, depthTempPersistRange, depthDisparityRange
from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame


class DepthAiGui(Cam):
    def __init__(self, gui: Gui | None, forceSize: tuple[int, int] | None = None, doMono: bool = True) -> None:
        self.gui: Gui | None = gui
        super().__init__(forceSize, doMono)

        elem: list = []
        elem.append([E(eT.TEXT, 'Exposure '),
                     E(eT.SLDR, 'Exposure',         super().setExposure,            0,     exposureRange,        500),
                     E(eT.TEXT, 'Iso      '),
                     E(eT.SLDR, 'Iso',              super().setIso,                 0,          isoRange,         50)])
        elem.append([E(eT.TEXT, 'Focus    '),
                     E(eT.SLDR, 'Focus',            super().setFocus,               0,        focusRange,          3),
                     E(eT.TEXT, 'W Balance'),
                     E(eT.SLDR, 'W Balance',        super().setWhiteBalance,        0, whiteBalanceRange,        200)])
        elem.append([E(eT.TEXT, '         '),
                     E(eT.CHCK, 'Auto Exposure',    super().setAutoExposure,        True),
                     E(eT.CHCK, 'Auto Focus',       super().setAutoFocus,           True),
                     E(eT.CHCK, 'Auto W Balance',   super().setAutoWhiteBalance,    True)])
        elem.append([E(eT.TEXT, 'Preview'),
                     E(eT.CMBO, 'Preview',          super().setPreview,            PreviewTypeNames[0], PreviewTypeNames)])

        self.color_frame = Frame('CAMERA', elem, 230)

        elem: list = []
        elem.append([E(eT.TEXT, 'Decimation'),
                     E(eT.SLDR, 'Decimation',       super().setDepthDecimation,     depthDecimationRange[0],    depthDecimationRange,   1),
                     E(eT.TEXT, 'Speckle'),
                     E(eT.SLDR, 'Speckle',          super().setDepthSpeckle,        depthSpeckleRange[0],       depthSpeckleRange,      1),])
        elem.append([E(eT.TEXT, 'Range Min'),
                     E(eT.SLDR, 'Min Range',        super().setDepthTresholdMin,    depthTresholdRange[0],      depthTresholdRange,     50),
                     E(eT.TEXT, 'Max'),
                     E(eT.SLDR, 'Max Range',        super().setDepthTresholdMax,    depthTresholdRange[1],      depthTresholdRange,     50)])
        elem.append([E(eT.TEXT, 'Hole Size'),
                     E(eT.SLDR, 'Hole Size',        super().setDepthHoleFilling,    depthHoleFillingRange[0],   depthHoleFillingRange,  1),
                     E(eT.TEXT, 'Iterations'),
                     E(eT.SLDR, 'Iterations',       super().setDepthHoleIter,       depthHoleIterRange[0],      depthHoleIterRange,     1)])
        elem.append([E(eT.TEXT, 'Temp Persist'),
                     E(eT.SLDR, 'Persistence',      super().setDepthTempPersist,    depthTempPersistRange[0],   depthTempPersistRange,  1),
                     E(eT.TEXT, 'Disparity'),
                     E(eT.SLDR, 'Disparity',        super().setDepthDisparityShift, depthDisparityRange[0],     depthDisparityRange,  1)])
        self.stereo_frame = Frame('STEREO', elem, 230)


        self.prevautoExposure: bool      = self.autoExposure
        self.prevAutoFocus: bool         = self.autoFocus
        self.prevAutoWhiteBalance: bool  = self.autoWhiteBalance
        self.prevExposure: int           = self.exposure
        self.prevIso: int                = self.iso
        self.prevFocus: int              = self.focus
        self.prevWhiteBalance: int       = self.whiteBalance


    def updateControlValues(self, frame) -> None:
        super().updateControlValues(frame)
        if not self.gui or not self.gui.isRunning(): return
        if (self.prevautoExposure != self.autoExposure) :
            self.prevautoExposure = self.autoExposure
            self.gui.updateElement('Auto Exposure', self.autoExposure)

        if (self.prevAutoFocus != self.autoFocus) :
            self.prevAutoFocus = self.autoFocus
            self.gui.updateElement('Auto Focus', self.autoFocus)

        if (self.prevAutoWhiteBalance != self.autoWhiteBalance) :
            self.prevAutoWhiteBalance = self.autoWhiteBalance
            self.gui.updateElement('Auto W Balance', self.autoWhiteBalance)

        if self.autoExposure:
            if (self.prevExposure != self.exposure) :
                self.prevExposure = self.exposure
                self.gui.updateElement('Exposure', self.exposure)
            if (self.prevIso != self.iso) :
                self.prevIso = self.iso
                self.gui.updateElement('Iso', self.iso)

        if self.autoFocus:
            if (self.prevFocus != self.focus) :
                self.prevFocus = self.focus
                self.gui.updateElement('Focus', self.focus)

        if self.autoWhiteBalance:
            if (self.prevWhiteBalance != self.whiteBalance) :
                self.prevWhiteBalance = self.whiteBalance
                self.gui.updateElement('W Balance', self.whiteBalance)

    def get_gui_frame(self):
          return self.get_color_frame()

    def get_color_frame(self):
          return self.color_frame

    def get_stereo_frame(self):
          return self.stereo_frame