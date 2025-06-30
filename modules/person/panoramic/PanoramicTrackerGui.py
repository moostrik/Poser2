from __future__ import annotations

from modules.Settings import Settings

from modules.person.panoramic.PanoramicDefinitions import *

from modules.gui.PyReallySimpleGui import Gui as G, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame, BASEHEIGHT, ELEMHEIGHT

class PanoramicTrackerGui():
    def __init__(self, gui: G, manager, settings: Settings) -> None:
        self.gui: G = gui
        self.manager = manager

        elm: list = []
        elm.append([E(eT.TEXT, 'CAM 360  '),
                    E(eT.TEXT, 'fov'),
                    E(eT.SLDR, 'fov',               self.set_fov,               CAM_360_FOV,                [90,130],   0.5),
                    E(eT.TEXT, 'edge'),
                    E(eT.SLDR, 'edge',              self.set_edge_threshold,    CAM_360_EDGE_THRESHOLD,     [0,0.5],    0.1),
                    E(eT.TEXT, 'ovlp'),
                    E(eT.SLDR, 'ovlp',              self.set_overlap,           CAM_360_OVERLAP_EXPANSION,  [0,1.0],    0.1)])
        elm.append([E(eT.TEXT, 'PERSON   '),
                    E(eT.TEXT, 'roi'),
                    E(eT.SLDR, 'roi exp',           self.set_roi_expansion,     PERSON_ROI_EXPANSION,       [0,0.5],    0.1),
                    E(eT.TEXT, 'timeout'),
                    E(eT.SLDR, 'timeout',           self.set_timeout,           PERSON_TIMEOUT,             [0.5,2.0],  0.1)])

        gui_height = len(elm) * ELEMHEIGHT + BASEHEIGHT
        self.frame = Frame('PERSON ', elm, gui_height)


    # GUI FRAME
    def get_gui_frame(self):
          return self.frame

    def set_fov(self, value: float) -> None:
        self.manager.geometry.set_fov(value)

    def set_edge_threshold(self, value: float) -> None:
        self.manager.cam_360_edge_threshold = value

    def set_overlap(self, value: float) -> None:
        self.manager.cam_360_overlap_expansion = value

    def set_roi_expansion(self, value: float) -> None:
        self.manager.person_roi_expansion = value

    def set_timeout(self, value: float) -> None:
        self.manager.person_timeout = value