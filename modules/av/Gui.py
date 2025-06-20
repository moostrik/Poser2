from __future__ import annotations

from modules.Settings import Settings

from modules.av.Definitions import *
# from modules.av.Manager import Manager
from modules.av.CompTest import CompTest, TestPattern, TEST_PATTERN_NAMES

from modules.gui.PyReallySimpleGui import Gui as G, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame, BASEHEIGHT, ELEMHEIGHT

class Gui():
    def __init__(self, gui: G, manager) -> None:
        self.gui: G = gui
        from modules.av.Manager import Manager
        self.manager: Manager = manager

        elm: list = []
        elm.append([E(eT.CMBO, 'Pattern',       manager.comp_test.set_pattern,          TEST_PATTERN_NAMES[0],  TEST_PATTERN_NAMES,   expand=False),
                    E(eT.TEXT, 'White'),
                    E(eT.SLDR, 'white_test',    manager.comp_test.set_white_strength,   1., [0., 1.],   0.01)])
        elm.append([E(eT.TEXT, 'Blue'),
                    E(eT.SLDR, 'blue_test',     manager.comp_test.set_blue_strength,    1., [0., 1.],   0.01),
                    E(eT.TEXT, 'Phase'),
                    E(eT.SLDR, 'blue_phase',    manager.comp_test.set_blue_phase,       1., [0., 1.],   0.01)])
        elm.append([E(eT.TEXT, 'Pulse  '),
                    E(eT.TEXT, 'Speed'),
                    E(eT.SLDR, 'plse_speed',    manager.comp_test.set_pulse_speed,      1., [0., 2.],   0.1)])
        elm.append([E(eT.TEXT, 'Chase  '),
                    E(eT.TEXT, 'Speed'),
                    E(eT.SLDR, 'chse_speed',    manager.comp_test.set_chase_speed,      1., [-2., 2.],  0.1),
                    E(eT.TEXT, 'Amnt '),
                    E(eT.SLDR, 'chse_amount',   manager.comp_test.set_chase_amount,     1,  [1, 10],    1)])
        elm.append([E(eT.TEXT, 'Single '),
                    E(eT.TEXT, 'Speed'),
                    E(eT.SLDR, 'sngl_speed',    manager.comp_test.set_single_speed,     1., [-2., 2.],  0.1),
                    E(eT.TEXT, 'Amnt '),
                    E(eT.SLDR, 'sngl_amount',   manager.comp_test.set_single_amount,    1,  [1, 10],    1)])
        elm.append([E(eT.TEXT, 'Random '),
                    E(eT.TEXT, 'Speed'),
                    E(eT.SLDR, 'rndm_speed',    manager.comp_test.set_random_speed,     1., [0., 2.],   0.1)])

        gui_height = len(elm) * ELEMHEIGHT + BASEHEIGHT
        self.frame = Frame('COMP TEST', elm, gui_height)

    # GUI FRAME
    def get_gui_frame(self):
          return self.frame