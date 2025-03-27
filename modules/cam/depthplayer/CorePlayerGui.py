from modules.cam.depthplayer.CorePlayer import *
from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame

class CorePlayerGui(CorePlayer):
    def __init__(self, gui: Gui | None, modelPath:str, fps: int = 30, doColor: bool = True, doStereo: bool = True, doPerson: bool = True, lowres: bool = False, showStereo: bool = False) -> None:
        self.gui: Gui | None = gui
        super().__init__(modelPath, fps, doColor, doStereo, doPerson, lowres, showStereo)

        elem: list = []
        elem.append([E(eT.TEXT, 'Depth Min '),
                     E(eT.SLDR, 'S_Min'+id,             super().set_depth_treshold_min,     STEREO_DEPTH_RANGE[0],  STEREO_DEPTH_RANGE, gsfr(STEREO_DEPTH_RANGE)),
                     E(eT.TEXT, 'Max'),
                     E(eT.SLDR, 'S_Max'+id,             super().set_depth_treshold_max,     STEREO_DEPTH_RANGE[0],  STEREO_DEPTH_RANGE, gsfr(STEREO_DEPTH_RANGE))])
        elem.append([E(eT.TEXT, 'Bright Min'),
                     E(eT.SLDR, 'S_BrighnessMin'+id,    super().set_stereo_min_brightness,  STEREO_BRIGHTNESS_RANGE[0], STEREO_BRIGHTNESS_RANGE, gsfr(STEREO_BRIGHTNESS_RANGE)),
                     E(eT.TEXT, 'Max'),
                     E(eT.SLDR, 'S_BrighnessMax'+id,    super().set_stereo_max_brightness,  STEREO_BRIGHTNESS_RANGE[0], STEREO_BRIGHTNESS_RANGE, gsfr(STEREO_BRIGHTNESS_RANGE))])