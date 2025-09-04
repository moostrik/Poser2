from __future__ import annotations


from modules.av.Definitions import *
# from modules.av.Manager import Manager
from modules.av.CompTest import TEST_PATTERN_NAMES

from modules.gui.PyReallySimpleGui import Gui as G, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame, BASEHEIGHT, ELEMHEIGHT

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.av.Manager import Manager

class Gui():
    def __init__(self, gui: G, manager: 'Manager') -> None:
        self.gui: G = gui
        self.manager: Manager = manager
        
        
        elm: list = []
        elm.append([E(eT.TEXT, '    '),
                    E(eT.CHCK, 'Show Overlay',  manager.comp.show_overlay),
                    E(eT.BTTN, 'Reset',         manager.comp.reset, expand=False),
                    E(eT.TEXT, 'FPS'),
                    E(eT.SLDR, 'avFPS',         None,                                   0,  [0,60],   0.1)])
        elm.append([E(eT.TEXT, 'Smth'),
                    E(eT.SLDR, 'Smooth',        manager.comp.set_smooth_alpha,          0.5, [0., 1.],  0.01)])
        elm.append([E(eT.TEXT, 'Void'),
                    E(eT.SLDR, 'Void',          manager.comp.set_void_width,            0.05, [0., 1.],  0.01),
                    E(eT.TEXT, 'Patt'),
                    E(eT.SLDR, 'Patt',          manager.comp.set_pattern_width,         0.2,  [0., 1.],  0.01)])
        elm.append([E(eT.TEXT, '  A0'),
                    E(eT.SLDR, 'A0',            manager.comp.set_input_A0,              0.05, [0., 1.],  0.01),
                    E(eT.TEXT, '  A1'),
                    E(eT.SLDR, 'A1',            manager.comp.set_input_A1,              0.2,  [0., 1.],  0.01)])
        elm.append([E(eT.TEXT, '  B0'),
                    E(eT.SLDR, 'B0',            manager.comp.set_input_B0,              0.05, [0., 1.],  0.01),
                    E(eT.TEXT, '  B1'),
                    E(eT.SLDR, 'B1',            manager.comp.set_input_B1,              0.2,  [0., 1.],  0.01)])
        gui_height = len(elm) * ELEMHEIGHT + BASEHEIGHT
        self.frame = Frame('COMP TEST', elm, gui_height)

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
        elm.append([E(eT.TEXT, 'Width    '),
                    E(eT.TEXT, 'W'),
                    E(eT.SLDR, 'W_width',       manager.comp_test.set_white_width,      0.5, [0., 1.],  0.01),
                    E(eT.TEXT, 'B'),
                    E(eT.SLDR, 'B_width',       manager.comp_test.set_blue_width,       0.5, [0., 1.],  0.01)])
        elm.append([E(eT.TEXT, 'Amount   '),
                    E(eT.TEXT, 'W'),
                    E(eT.SLDR, 'W_amount',      manager.comp_test.set_white_amount,     36,  [1, 360],  1),
                    E(eT.TEXT, 'B'),
                    E(eT.SLDR, 'B_amount',      manager.comp_test.set_blue_amount,      36,  [1, 360],  1)])

        gui_height = len(elm) * ELEMHEIGHT + BASEHEIGHT
        self.test_frame = Frame('COMP TEST', elm, gui_height)

    # GUI FRAME
    def get_gui_frame(self):
          return self.frame
    def get_gui_test_frame(self):
          return self.test_frame

    def reset(self) -> None:
        self.manager.comp_test.reset()
        self.update()

    def white_to_blue(self) -> None:
        self.manager.comp_test.white_to_blue()
        self.update()

    def update(self) -> None:
        if not self.gui:
            return
        self.gui.updateElement('avFPS', self.manager.FPS.get_fps())
