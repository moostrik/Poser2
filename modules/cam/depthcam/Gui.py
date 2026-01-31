import math

from modules.cam.Config import Config
from modules.cam.depthcam.CoreSettings import CoreSettings
from modules.cam.depthcam.Definitions import *
from modules.gui.PyReallySimpleGui import Gui as G, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame, BASEHEIGHT, ELEMHEIGHT

def closest_log_value(number: float) -> float:
    return 10 ** round(math.log10(number))

def get_steps_from_range(range: tuple[int, int]) -> float:
    roughSteps: float = float(range[1] - range[0]) / 100.0
    return closest_log_value(roughSteps)

def gsfr(range: tuple[int, int]) -> float:
    return get_steps_from_range(range)

class Gui():
    def __init__(self, gui: G, settings: CoreSettings, general_settings: Config) -> None:
        self.gui: G = gui
        self.settings: CoreSettings = settings

        self.id: str = self.settings.get_id_string()
        self.simulation: bool = general_settings.sim_enabled
        self.manual: bool = False if self.simulation else general_settings.manual
        self.color: bool = False if self.simulation else general_settings.color
        self.stereo: bool = general_settings.stereo
        self.ftn: list[str] = self.settings.get_frame_type_names()

        id: str = self.id
        elm: list = []
        elm.append([E(eT.CMBO, 'Preview'+id,           self.settings.set_preview,              self.ftn[1],            self.ftn,   expand=False),
                    E(eT.TEXT, '  FPS V'),
                    E(eT.SLDR, 'FPS_V'+id,             None,                               0,  [0,60],   0.1),
                    E(eT.TEXT, ' T'),
                    E(eT.SLDR, 'TPS'+id,               None,                               0,  [0,60],   0.1),
                    E(eT.TEXT, 'Num T'),
                    E(eT.SLDR, 'NumTracklets'+id,      None,                               0,  [0,6],    1)])

        if self.color:
            if self.manual:
                elm.append([E(eT.TEXT, 'Exposure  '),
                            E(eT.SLDR, 'C_Exposure'+id,        self.settings.set_color_exposure,       EXPOSURE_RANGE[0],      EXPOSURE_RANGE,     gsfr(EXPOSURE_RANGE)),
                            E(eT.TEXT, '       Iso'),
                            E(eT.SLDR, 'C_Iso'+id,             self.settings.set_color_iso,            ISO_RANGE[0],           ISO_RANGE,          gsfr(ISO_RANGE))])
                elm.append([E(eT.TEXT, 'Balance   '),
                            E(eT.SLDR, 'C_Balance'+id,         self.settings.set_color_balance,        BALANCE_RANGE[0],       BALANCE_RANGE,      gsfr(BALANCE_RANGE)),
                            E(eT.TEXT, '  Contrast'),
                            E(eT.SLDR, 'C_Contrast'+id,        self.settings.set_color_contrast,       CONTRAST_RANGE[0],      CONTRAST_RANGE,     gsfr(CONTRAST_RANGE))])
                elm.append([E(eT.TEXT, 'Brightness'),
                            E(eT.SLDR, 'C_Brightness'+id,      self.settings.set_color_brightness,     BRIGHTNESS_RANGE[0],    BRIGHTNESS_RANGE,   gsfr(BRIGHTNESS_RANGE)),
                            E(eT.TEXT, '   Denoise'),
                            E(eT.SLDR, 'C_Denoise'+id,         self.settings.set_color_denoise,        LUMA_DENOISE_RANGE[0],  LUMA_DENOISE_RANGE, gsfr(LUMA_DENOISE_RANGE))])
                elm.append([E(eT.TEXT, 'Saturation'),
                            E(eT.SLDR, 'C_Saturation'+id,      self.settings.set_color_saturation,     SATURATION_RANGE[0],    SATURATION_RANGE,   gsfr(SATURATION_RANGE)),
                            E(eT.TEXT, ' Sharpness'),
                            E(eT.SLDR, 'C_Sharpness'+id,       self.settings.set_color_sharpness,      SHARPNESS_RANGE[0],     SHARPNESS_RANGE,    gsfr(SHARPNESS_RANGE))])
                elm.append([E(eT.TEXT, 'Auto      '),
                            E(eT.CHCK, 'AutoExposure'+id,      self.settings.set_color_auto_exposure,  True),
                            E(eT.CHCK, 'AutoBalance'+id,       self.settings.set_color_auto_balance,   True)])
        if not self.color or self.stereo:
            if not self.simulation:
                elm.append([E(eT.TEXT, 'IR Grid   '),
                            E(eT.SLDR, 'L_Grid'+id,            self.set_ir_grid_light,                 0, [0,1], 0.05),
                            E(eT.TEXT, 'Fld'),
                            E(eT.SLDR, 'L_Flood'+id,           self.set_ir_flood_light,                0, [0,1], 0.05)])
            if self.manual:
                elm.append([E(eT.CHCK, 'M_AE_'+id,    self.settings.set_mono_auto_exposure,  True),
                            E(eT.SLDR, 'M_Exposure'+id,        self.settings.set_mono_exposure,        EXPOSURE_RANGE[0],      EXPOSURE_RANGE,     gsfr(EXPOSURE_RANGE)),
                            E(eT.TEXT, 'Iso'),
                            E(eT.SLDR, 'M_Iso'+id,             self.settings.set_mono_iso,             ISO_RANGE[0],           ISO_RANGE,          gsfr(ISO_RANGE))])
                # elm.append([E(eT.CHCK, 'M_AE'+id,    self.settings.set_mono_auto_exposure,  True)])
        if self.stereo:
            elm.append([E(eT.TEXT, 'Depth Min '),
                        E(eT.SLDR, 'S_Min'+id,             self.settings.set_depth_treshold_min,   STEREO_DEPTH_RANGE[0],  STEREO_DEPTH_RANGE, gsfr(STEREO_DEPTH_RANGE)),
                        E(eT.TEXT, 'Max'),
                        E(eT.SLDR, 'S_Max'+id,             self.settings.set_depth_treshold_max,   STEREO_DEPTH_RANGE[0],  STEREO_DEPTH_RANGE, gsfr(STEREO_DEPTH_RANGE))])
            elm.append([E(eT.TEXT, 'Bright Min'),
                        E(eT.SLDR, 'S_BrighnessMin'+id,    self.settings.set_stereo_min_brightness,STEREO_BRIGHTNESS_RANGE[0], STEREO_BRIGHTNESS_RANGE, gsfr(STEREO_BRIGHTNESS_RANGE)),
                        E(eT.TEXT, 'Max'),
                        E(eT.SLDR, 'S_BrighnessMax'+id,    self.settings.set_stereo_max_brightness,STEREO_BRIGHTNESS_RANGE[0], STEREO_BRIGHTNESS_RANGE, gsfr(STEREO_BRIGHTNESS_RANGE))])
            elm.append([E(eT.TEXT, 'Filter    '),
                        E(eT.CMBO, 'Median'+id,            self.settings.set_stereo_median_filter, STEREO_FILTER_NAMES[0], STEREO_FILTER_NAMES, expand=False)])

        gui_height = len(elm) * ELEMHEIGHT + BASEHEIGHT
        self.frame = Frame('CAMERA ' + self.id, elm, gui_height)

        self.prev_color_auto_exposure: bool =   self.settings.color_auto_exposure
        self.prev_color_exposure: int =         self.settings.color_exposure
        self.prev_color_iso: int =              self.settings.color_iso
        self.prev_color_auto_balance: bool =    self.settings.color_auto_balance
        self.prev_color_white_balance: int =    self.settings.color_balance
        self.prev_mono_auto_exposure: bool =    self.settings.mono_auto_exposure
        self.prev_mono_exposure: int =          self.settings.mono_exposure
        self.prev_mono_iso: int =               self.settings.mono_iso

    # COLOR
    def update_from_frame(self) -> None:
        if not self.gui.isRunning(): return

        # fps
        self.gui.updateElement('FPS_V'+self.id, self.settings.get_fps(FrameType.VIDEO))

        if not self.manual:
            return

        if self.color:
            if (self.prev_color_auto_exposure != self.settings.color_auto_exposure) :
                self.prev_color_auto_exposure = self.settings.color_auto_exposure
                self.gui.updateElement('AutoExposure'+self.id, self.settings.color_auto_exposure)

            if (self.prev_color_auto_balance != self.settings.color_auto_balance) :
                self.prev_color_auto_balance = self.settings.color_auto_balance
                self.gui.updateElement('AutoBalance'+self.id, self.settings.color_auto_balance)

            if self.settings.color_auto_exposure:
                if (self.prev_color_exposure != self.settings.color_exposure) :
                    self.prev_color_exposure = self.settings.color_exposure
                    self.gui.updateElement('C_Exposure'+self.id, self.settings.color_exposure)
                if (self.prev_color_iso != self.settings.color_iso) :
                    self.prev_color_iso = self.settings.color_iso
                    self.gui.updateElement('C_Iso'+self.id, self.settings.color_iso)

            if self.settings.color_auto_balance:
                if (self.prev_color_white_balance != self.settings.color_balance) :
                    self.prev_color_white_balance = self.settings.color_balance
                    self.gui.updateElement('C_Balance'+self.id, self.settings.color_balance)

        if not self.color or self.stereo:
            if (self.prev_mono_auto_exposure != self.settings.mono_auto_exposure) :
                self.prev_mono_auto_exposure = self.settings.mono_auto_exposure
                self.gui.updateElement('M_AE_'+self.id, self.settings.mono_auto_exposure)

            if self.settings.mono_auto_exposure:
                if (self.prev_mono_exposure != self.settings.mono_exposure) :
                    self.prev_mono_exposure = self.settings.mono_exposure
                    self.gui.updateElement('M_Exposure'+self.id, self.settings.mono_exposure)
                if (self.prev_mono_iso != self.settings.mono_iso) :
                    self.prev_mono_iso = self.settings.mono_iso
                    self.gui.updateElement('M_Iso'+self.id, self.settings.mono_iso)

    def update_from_tracker(self) -> None:
        if not self.gui.isRunning(): return
        self.gui.updateElement('NumTracklets'+self.id,  self.settings.get_num_tracklets())
        self.gui.updateElement('TPS'+self.id, self.settings.get_tps())
        pass

    # IR
    def set_ir_grid_light(self, value: float) -> None:
        self.settings.set_ir_grid_light(value)
        if not self.gui or not self.gui.isRunning(): return
        if value > 0.0:
            self.gui.updateElement('L_Flood'+self.id, 0.0)
            self.settings.set_ir_flood_light(0.0)

    def set_ir_flood_light(self, value: float) -> None:
        self.settings.set_ir_flood_light(value)
        if not self.gui or not self.gui.isRunning(): return
        if value > 0.0:
            self.gui.updateElement('L_Grid'+self.id, 0.0)
            self.settings.set_ir_grid_light(0.0)

    # GUI FRAME
    def get_gui_frame(self):
          return self.frame

    def gui_check(self) -> None:
        if not self.gui.isRunning(): return
        e: str = 'Preview'+self.id
        if not self.gui.getStringValue(e) in self.ftn:
            p: str = self.ftn[1]
            self.gui.updateElement(e, p)
            self.settings.set_preview(p)
