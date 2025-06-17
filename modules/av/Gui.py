from __future__ import annotations

from modules.Settings import Settings

from modules.av.Definitions import *

from modules.gui.PyReallySimpleGui import Gui as G, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame, BASEHEIGHT, ELEMHEIGHT

class Gui():
    def __init__(self, gui: G, manager, settings: Settings) -> None:
        self.gui: G = gui
        self.manager = manager

        elm: list = []
        elm.append([E(eT.TEXT, 'CAM 360  ')])
        elm.append([E(eT.TEXT, 'PERSON   ')])

        gui_height = len(elm) * ELEMHEIGHT + BASEHEIGHT
        self.frame = Frame('COMPOSTION', elm, gui_height)
