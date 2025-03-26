import math

from modules.cam.DepthAi.CoreSettings import *
from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame

def closest_log_value(number: float) -> float:
    return 10 ** round(math.log10(number))

def get_steps_from_range(range: tuple[int, int]) -> float:
    roughSteps: float = float(range[1] - range[0]) / 100.0
    return closest_log_value(roughSteps)

def gsfr(range: tuple[int, int]) -> float:
    return get_steps_from_range(range)

class DepthAiGui(DepthAiSettings):
    def __init__(self, gui: Gui | None, modelPath:str, fps: int = 30, doColor: bool = True, doStereo: bool = True, doPerson: bool = True, lowres: bool = False, showStereo: bool = False) -> None:
        self.gui: Gui | None = gui
        super().__init__(modelPath, fps, doColor, doStereo, doPerson, lowres, showStereo)

        id: str = self.id_string
        elem: list = []
        elem.append([E(eT.TEXT, 'Exposure  '),
                     E(eT.SLDR, 'C_Exposure'+id,        super().set_color_exposure,         EXPOSURE_RANGE[0],      EXPOSURE_RANGE,     gsfr(EXPOSURE_RANGE)),
                     E(eT.TEXT, '       Iso'),
                     E(eT.SLDR, 'C_Iso'+id,             super().set_color_iso,              ISO_RANGE[0],           ISO_RANGE,          gsfr(ISO_RANGE))])
        elem.append([E(eT.TEXT, 'Balance   '),
                     E(eT.SLDR, 'C_Balance'+id,         super().set_color_balance,          BALANCE_RANGE[0],       BALANCE_RANGE,      gsfr(BALANCE_RANGE)),
                     E(eT.TEXT, '  Contrast'),
                     E(eT.SLDR, 'C_Contrast'+id,        super().set_color_contrast,         CONTRAST_RANGE[0],      CONTRAST_RANGE,     gsfr(CONTRAST_RANGE))])
        elem.append([E(eT.TEXT, 'Brightness'),
                     E(eT.SLDR, 'C_Brightness'+id,      super().set_color_brightness,       BRIGHTNESS_RANGE[0],    BRIGHTNESS_RANGE,   gsfr(BRIGHTNESS_RANGE)),
                     E(eT.TEXT, '   Denoise'),
                     E(eT.SLDR, 'C_Denoise'+id,         super().set_color_denoise,          LUMA_DENOISE_RANGE[0],  LUMA_DENOISE_RANGE, gsfr(LUMA_DENOISE_RANGE))])
        elem.append([E(eT.TEXT, 'Saturation'),
                     E(eT.SLDR, 'C_Saturation'+id,      super().set_color_saturation,       SATURATION_RANGE[0],    SATURATION_RANGE,   gsfr(SATURATION_RANGE)),
                     E(eT.TEXT, ' Sharpness'),
                     E(eT.SLDR, 'C_Sharpness'+id,       super().set_color_sharpness,        SHARPNESS_RANGE[0],     SHARPNESS_RANGE,    gsfr(SHARPNESS_RANGE))])
        elem.append([E(eT.CMBO, 'Preview'+id,           super().set_preview,                self.get_frame_type_names()[0],  self.get_frame_type_names(),   expand=False),
                     E(eT.CHCK, 'AutoExposure'+id,      super().set_color_auto_exposure,    True),
                     E(eT.CHCK, 'AutoBalance'+id,       super().set_color_auto_balance,     True),
                     E(eT.SLDR, 'FPS'+id,               None,                               0,    [0,60],   0.1)])
        elem.append([ E(eT.SLDR, 'NumTracklets'+id,     None,                               0,    [0,6],    1),
                     E(eT.SLDR, 'TPS'+id,               None,                               0,    [0,60],   0.1)])


        self.color_frame = Frame('CAMERA COLOR', elem, 240)

        elem: list = []
        elem.append([E(eT.TEXT, 'Exposure  '),
                     E(eT.SLDR, 'M_Exposure'+id,        super().set_mono_exposure,          EXPOSURE_RANGE[0],      EXPOSURE_RANGE,     gsfr(EXPOSURE_RANGE)),
                     E(eT.TEXT, 'Iso'),
                     E(eT.SLDR, 'M_Iso'+id,             super().set_mono_iso,               ISO_RANGE[0],           ISO_RANGE,          gsfr(ISO_RANGE))])
        elem.append([E(eT.TEXT, 'Depth Min '),
                     E(eT.SLDR, 'S_Min'+id,             super().set_depth_treshold_min,     STEREO_DEPTH_RANGE[0],  STEREO_DEPTH_RANGE, gsfr(STEREO_DEPTH_RANGE)),
                     E(eT.TEXT, 'Max'),
                     E(eT.SLDR, 'S_Max'+id,             super().set_depth_treshold_max,     STEREO_DEPTH_RANGE[0],  STEREO_DEPTH_RANGE, gsfr(STEREO_DEPTH_RANGE))])
        elem.append([E(eT.TEXT, 'Bright Min'),
                     E(eT.SLDR, 'S_BrighnessMin'+id,    super().set_stereo_min_brightness,  STEREO_BRIGHTNESS_RANGE[0], STEREO_BRIGHTNESS_RANGE, gsfr(STEREO_BRIGHTNESS_RANGE)),
                     E(eT.TEXT, 'Max'),
                     E(eT.SLDR, 'S_BrighnessMax'+id,    super().set_stereo_max_brightness,  STEREO_BRIGHTNESS_RANGE[0], STEREO_BRIGHTNESS_RANGE, gsfr(STEREO_BRIGHTNESS_RANGE))])
        elem.append([E(eT.TEXT, 'IR Grid   '),
                     E(eT.SLDR, 'L_Grid'+id,            self.set_ir_grid_light,             0, [0,1], 0.05),
                     E(eT.TEXT, 'Fld'),
                     E(eT.SLDR, 'L_Flood'+id,           self.set_ir_flood_light,            0, [0,1], 0.05)])
        elem.append([E(eT.TEXT, 'Filter    '),
                     E(eT.CMBO, 'Median'+id,            super().set_stereo_median_filter,   STEREO_FILTER_NAMES[0], STEREO_FILTER_NAMES, expand=False),
                     E(eT.TEXT, '              '),
                     E(eT.CHCK, 'M_AutoExposure'+id,    super().set_mono_auto_exposure,     False)])

        self.depth_frame = Frame('CAMERA DEPTH', elem, 200)

        self.prev_color_auto_exposure: bool =   self.color_auto_exposure
        self.prev_color_exposure: int =         self.color_exposure
        self.prev_color_iso: int =              self.color_iso
        self.prev_color_auto_balance: bool =    self.color_auto_balance
        self.prev_color_white_balance: int =    self.color_balance
        self.prev_mono_auto_exposure: bool =    self.mono_auto_exposure
        self.prev_mono_exposure: int =          self.mono_exposure
        self.prev_mono_iso: int =               self.mono_iso

    # COLOR
    def _update_color_control(self, frame) -> None: #override
        super()._update_color_control(frame)
        if not self.gui or not self.gui.isRunning(): return
        id: str = self.id_string

        if (self.prev_color_auto_exposure != self.color_auto_exposure) :
            self.prev_color_auto_exposure = self.color_auto_exposure
            self.gui.updateElement('AutoExposure'+id, self.color_auto_exposure)

        if (self.prev_color_auto_balance != self.color_auto_balance) :
            self.prev_color_auto_balance = self.color_auto_balance
            self.gui.updateElement('AutoBalance'+id, self.color_auto_balance)

        if self.color_auto_exposure:
            if (self.prev_color_exposure != self.color_exposure) :
                self.prev_color_exposure = self.color_exposure
                self.gui.updateElement('C_Exposure'+id, self.color_exposure)
            if (self.prev_color_iso != self.color_iso) :
                self.prev_color_iso = self.color_iso
                self.gui.updateElement('C_Iso'+id, self.color_iso)

        if self.color_auto_balance:
            if (self.prev_color_white_balance != self.color_balance) :
                self.prev_color_white_balance = self.color_balance
                self.gui.updateElement('C_Balance'+id, self.color_balance)


    # MONO
    def _update_mono_control(self, frame) -> None: #override
        super()._update_mono_control(frame)
        if not self.gui or not self.gui.isRunning(): return
        id: str = self.id_string

        if (self.prev_mono_auto_exposure != self.mono_auto_exposure) :
            self.prev_mono_auto_exposure = self.mono_auto_exposure
            self.gui.updateElement('M_AutoExposure'+id, self.mono_auto_exposure)

        if self.mono_auto_exposure:
            if (self.prev_mono_exposure != self.mono_exposure) :
                self.prev_mono_exposure = self.mono_exposure
                self.gui.updateElement('M_Exposure'+id, self.mono_exposure)
            if (self.prev_mono_iso != self.mono_iso) :
                self.prev_mono_iso = self.mono_iso
                self.gui.updateElement('M_Iso'+id, self.mono_iso)

    # IR
    def set_ir_grid_light(self, value: float) -> None: #override
        super().set_ir_grid_light(value)
        if not self.gui or not self.gui.isRunning(): return
        if value > 0.0:
            self.gui.updateElement('L_Flood'+self.id_string, 0.0)
            super().set_ir_flood_light(0.0)

    def set_ir_flood_light(self, value: float) -> None: #override
        super().set_ir_flood_light(value)
        if not self.gui or not self.gui.isRunning(): return
        if value > 0.0:
            self.gui.updateElement('L_Grid'+self.id_string, 0.0)
            super().set_ir_grid_light(0.0)

    # FPS
    def _update_fps(self) -> None: #override
        super()._update_fps()
        if not self.gui or not self.gui.isRunning(): return
        self.gui.updateElement('FPS'+self.id_string, self.get_fps())
        self.gui.updateElement('TPS'+self.id_string, self.get_tps())

        # in a perfect world, this would have its own defintion
        self.gui.updateElement('NumTracklets'+self.id_string,  self.num_tracklets)

    # GUI FRAME
    def get_gui_frame(self):
          return self.get_gui_color_frame()

    def get_gui_color_frame(self):
          return self.color_frame

    def get_gui_depth_frame(self):
        return self.depth_frame

    def gui_check(self) -> None:
        if not self.gui: return
        e: str = 'Preview'+self.id_string
        if not self.gui.getStringValue(e) in self.get_frame_type_names():
            p: str = self.get_frame_type_names()[1]
            self.gui.updateElement(e, p)
            self.set_preview(p)
