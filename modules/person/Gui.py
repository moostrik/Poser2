from __future__ import annotations

from modules.Settings import Settings

from modules.person.CircularCoordinates import CircularCoordinates
from modules.person.Definitions import *

from modules.gui.PyReallySimpleGui import Gui as G, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame

class Gui():
    def __init__(self, gui: G, manager, settings: Settings) -> None:
        self.gui: G = gui
        self.manager = manager

        elm: list = []
        elm.append([E(eT.TEXT, '  FOV'),
                    E(eT.SLDR, 'fov',               self.manager.circular_coordinates.set_fov,              CAMERA_FOV,     [90,130],   0.5),
                    E(eT.TEXT, 'Angle'),
                    E(eT.SLDR, 'angle range',       self.manager.circular_coordinates.set_angle_range,      ANGLE_RANGE,    [1,11],     0.5),
                    E(eT.TEXT, 'Vrtcl'),
                    E(eT.SLDR, 'vrtcl range',       self.manager.circular_coordinates.set_vertical_range,   VERTICAL_RANGE, [0,0.1],    0.01),
                    E(eT.TEXT, ' Size'),
                    E(eT.SLDR, 'size range',        self.manager.circular_coordinates.set_size_range,       SIZE_RANGE,     [0,0.1],    0.01)])

        gui_height = len(elm) * 31 + 50
        self.frame = Frame('PERSON ', elm, gui_height)


    # GUI FRAME
    def get_gui_frame(self):
          return self.frame

