from __future__ import annotations


from modules.av.Definitions import *
# from modules.av.Manager import Manager
from modules.av.CompTest import TEST_PATTERN_NAMES

from modules.av.Definitions import CompSettings
from modules.gui.PyReallySimpleGui import Gui as G, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame, BASEHEIGHT, ELEMHEIGHT

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from modules.av.Manager import Manager

class Gui():
    def __init__(self, gui: G, manager: 'Manager', comp_settings: CompSettings) -> None:
        self.gui: G = gui
        self.manager: Manager = manager
        self.comp_settings: CompSettings = comp_settings

        elm: list = []
        elm.append([E(eT.TEXT, '    '),
                    E(eT.CHCK, 'Show Overlay',  self.set_use_void,          comp_settings.use_void),
                    E(eT.BTTN, 'Reset',         manager.comp.reset,             expand=False),
                    E(eT.TEXT, 'FPS'),
                    E(eT.SLDR, 'avFPS',         None,                       0,  [0,60],   0.1)])
        elm.append([E(eT.TEXT, 'Pose'),
                    E(eT.TEXT, 'Smoot'),
                    E(eT.SLDR, 'Smooth',        self.set_smoothness,        comp_settings.smoothness,       [0., 1.],  0.01),
                    E(eT.TEXT, 'Resp'),
                    E(eT.SLDR, 'Resp',          self.set_responsiveness,    comp_settings.responsiveness,   [0., 1.],  0.01)])
        elm.append([E(eT.TEXT, 'Void'),
                    E(eT.TEXT, 'Width'),
                    E(eT.SLDR, 'VoidWidth',     self.set_void_width,        comp_settings.void_width,       [1,  20],  0.5),
                    E(eT.TEXT, 'Edge'),
                    E(eT.SLDR, 'VoidEdge',      self.set_void_edge,         comp_settings.void_edge,        [0., 10],  0.5)])
        elm.append([E(eT.TEXT, 'Patt'),
                    E(eT.TEXT, 'Width'),
                    E(eT.SLDR, 'PattWidth',     self.set_pattern_width,     comp_settings.pattern_width,    [0.,360],  1.0),
                    E(eT.TEXT, 'Edge'),
                    E(eT.SLDR, 'PattEdge',      self.set_pattern_edge,      comp_settings.pattern_edge,     [0., 30],  0.5)])
        elm.append([E(eT.TEXT, '    '),
                    E(eT.TEXT, 'Speed'),
                    E(eT.SLDR, 'PattSpeed',     self.set_pattern_speed,     comp_settings.pattern_speed,    [0.,360],  1.0),
                    E(eT.TEXT, 'Shrp'),
                    E(eT.SLDR, 'PattSharp',     self.set_pattern_sharpness, comp_settings.pattern_sharpness,     [0., 30],  0.5)])
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


    def set_use_void(self, value) -> None:
        self.comp_settings.use_void = value
        self.manager.comp.update_settings()

    def set_smoothness(self, value) -> None:
        self.comp_settings.smoothness = value
        self.manager.comp.update_settings()

    def set_responsiveness(self, value) -> None:
        self.comp_settings.responsiveness = value
        self.manager.comp.update_settings()

    def set_void_width(self, value) -> None:
        self.comp_settings.void_width = value / 360
        self.manager.comp.update_settings()

    def set_void_edge(self, value) -> None:
        self.comp_settings.void_edge = value / 360
        self.manager.comp.update_settings()

    def set_pattern_width(self, value) -> None:
        self.comp_settings.pattern_width = value / 360
        self.manager.comp.update_settings()

    def set_pattern_edge(self, value) -> None:
        self.comp_settings.pattern_edge = value / 360
        self.manager.comp.update_settings()

    def set_pattern_speed(self, value) -> None:
        self.comp_settings.pattern_speed = value / 10
        self.manager.comp.update_settings()

    def set_pattern_sharpness(self, value) -> None:
        self.comp_settings.pattern_sharpness = value
        self.manager.comp.update_settings()
