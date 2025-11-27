from __future__ import annotations

from modules.Settings import Settings

from modules.tracker.panoramic.PanoramicDefinitions import *

from modules.gui.PyReallySimpleGui import Gui as G, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame, BASEHEIGHT, ELEMHEIGHT

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.tracker.onepercam.OnePerCamTracker import OnePerCamTracker

class OnePerCamTrackerGui():
    def __init__(self, gui: G, tracker: "OnePerCamTracker") -> None:
        self.gui: G = gui
        self.tracker: OnePerCamTracker = tracker

        elm: list = []
        elm.append([E(eT.TEXT, 'TIME     '),
                    E(eT.TEXT, 'min age'),
                    E(eT.SLDR, 'min_age',       self.set_min_age,                   5,      [0,9],      1),
                    E(eT.TEXT, 'timeout'),
                    E(eT.SLDR, 'timeout',       self.set_timeout,                   1.0,    [1.0,5.0],  0.1)])
        elm.append([E(eT.TEXT, 'ADD      '),
                    E(eT.TEXT, 'centre'),
                    E(eT.SLDR, 'A_centre',      self.set_add_centre_threshold,      0.15,   [0.0,1.0],  0.05),
                    E(eT.TEXT, 'height'),
                    E(eT.SLDR, 'A_height',      self.set_add_height_threshold,      0.3,    [0.0,1.0],  0.05)])
        elm.append([E(eT.TEXT, '         '),
                    E(eT.TEXT, 'bottom'),
                    E(eT.SLDR, 'A_bottom',      self.set_add_bottom_threshold,      0.25,    [0.0,1.0],  0.05)])
        elm.append([E(eT.TEXT, 'UPDATE   '),
                    E(eT.TEXT, 'centre'),
                    E(eT.SLDR, 'U_centre',      self.set_update_centre_threshold,   0.3,    [0.0,1.0],  0.05),
                    E(eT.TEXT, 'height'),
                    E(eT.SLDR, 'U_height',      self.set_update_height_treshold,    0.25,   [0.0,1.0],  0.05)])

        gui_height: int = len(elm) * ELEMHEIGHT + BASEHEIGHT
        self.frame = Frame('SCREEN TRACKER ', elm, gui_height)

    # GUI FRAME
    def get_gui_frame(self):
          return self.frame

    def set_min_age(self, value: float) -> None:
        self.tracker.tracklet_min_age = int(value)

    def set_timeout(self, value: float) -> None:
        self.tracker.timeout = value

    def set_add_centre_threshold(self, value: float) -> None:
        self.tracker.add_centre_threshold = value

    def set_add_height_threshold(self, value: float) -> None:
        self.tracker.add_height_threshold = value

    def set_add_bottom_threshold(self, value: float) -> None:
        self.tracker.add_bottom_threshold = value

    def set_update_centre_threshold(self, value: float) -> None:
        self.tracker.update_centre_threshold = value

    def set_update_height_treshold(self, value: float) -> None:
        self.tracker.update_height_treshold = value
