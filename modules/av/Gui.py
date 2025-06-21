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
        elm.append([E(eT.TEXT, 'Pattern  '),
                    E(eT.CMBO, 'Pattern',       manager.comp_test.set_pattern,          TEST_PATTERN_NAMES[0],  TEST_PATTERN_NAMES,   expand=True),
                    E(eT.BTTN, 'Reset',         self.reset),
                    E(eT.BTTN, 'Copy Blue',     self.white_to_blue)])
        elm.append([E(eT.TEXT, 'Strength '),
                    E(eT.TEXT, 'W'),
                    E(eT.SLDR, 'W_test',        manager.comp_test.set_white_strength,   0.5, [0., 1.],  0.01),
                    E(eT.TEXT, 'B'),
                    E(eT.SLDR, 'B_test',        manager.comp_test.set_blue_strength,    0.5, [0., 1.],  0.01)])
        elm.append([E(eT.TEXT, 'Speed    '),
                    E(eT.TEXT, 'W'),
                    E(eT.SLDR, 'W_speed',       manager.comp_test.set_white_speed,      0.5, [-1., 1.], 0.01),
                    E(eT.TEXT, 'B'),
                    E(eT.SLDR, 'B_speed',       manager.comp_test.set_blue_speed,       0.5, [-1., 1.], 0.01)])
        elm.append([E(eT.TEXT, 'Phase    '),
                    E(eT.TEXT, 'W'),
                    E(eT.SLDR, 'W_phase',       manager.comp_test.set_white_phase,      0.0, [0., 1.],  0.01),
                    E(eT.TEXT, 'B'),
                    E(eT.SLDR, 'B_phase',       manager.comp_test.set_blue_phase,       0.5, [0., 1.],  0.01)])
        elm.append([E(eT.TEXT, 'Amount   '),
                    E(eT.TEXT, 'W'),
                    E(eT.SLDR, 'W_amount',      manager.comp_test.set_white_amount,     36,  [1, 360],  1),
                    E(eT.TEXT, 'B'),
                    E(eT.SLDR, 'B_amount',      manager.comp_test.set_blue_amount,      36,  [1, 360],  1)])
        elm.append([E(eT.TEXT, 'Width    '),
                    E(eT.TEXT, 'W'),
                    E(eT.SLDR, 'W_width',       manager.comp_test.set_white_width,      3,   [1, 36],   0.01),
                    E(eT.TEXT, 'B'),
                    E(eT.SLDR, 'B_width',       manager.comp_test.set_blue_width,       3,   [1, 36],   0.01)])

        gui_height = len(elm) * ELEMHEIGHT + BASEHEIGHT
        self.frame = Frame('COMP TEST', elm, gui_height)

    # GUI FRAME
    def get_gui_frame(self):
          return self.frame

    def reset(self) -> None:
        self.manager.comp_test.reset()
        self.update()

    def white_to_blue(self) -> None:
        self.manager.comp_test.white_to_blue()
        self.update()

    def update(self) -> None:
        if not self.gui:
            return
        self.gui.updateElement('W_test', self.manager.comp_test.WP.strength)
        self.gui.updateElement('B_test', self.manager.comp_test.BP.strength)
        self.gui.updateElement('W_speed', self.manager.comp_test.WP.speed)
        self.gui.updateElement('B_speed', self.manager.comp_test.BP.speed)
        self.gui.updateElement('W_phase', self.manager.comp_test.WP.phase)
        self.gui.updateElement('B_phase', self.manager.comp_test.BP.phase)
        self.gui.updateElement('W_amount', self.manager.comp_test.WP.amount)
        self.gui.updateElement('B_amount', self.manager.comp_test.BP.amount)
        self.gui.updateElement('W_width', self.manager.comp_test.WP.width)
        self.gui.updateElement('B_width', self.manager.comp_test.BP.width)
