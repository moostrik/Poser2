from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame
from modules.render.Render import Render
from modules.render.Sequencer import SequenceType, SeqenceMessage, SequenceTypeNames
import pytweening as easing
from typing import Callable

EasingNames: list[str] = ['Linear',
                          'Sine In', 'Sine Out', 'Sine In Out',
                          'Quad In', 'Quad Out', 'Quad In Out',
                          'Cubic In', 'Cubic Out', 'Cubic In Out',
                          'Quart In', 'Quart Out', 'Quart In Out',
                          'Quint In', 'Quint Out', 'Quint In Out']

class RenderGui(Render):
    def __init__(self, gui: Gui, composition_width: int, composition_height: int, window_width: int, window_height: int, window_name: str, fullscreen=False, v_sync=True, stretch=False) -> None:
        super(RenderGui, self).__init__(composition_width, composition_height, window_width, window_height, window_name, fullscreen, v_sync, stretch)

        self.gui: Gui = gui
        self.trail_active: bool = True
        self.trail_gui: float = 0.996
        elem: list = []
        elem.append([E(eT.TEXT, 'VIEW     '),
                     E(eT.CHCK, 'Fullscreen',   self.set_fullscreen,                fullscreen),
                     E(eT.CHCK, 'Show Camera',  self.gui_set_draw_cam,              False),
                     E(eT.CHCK, 'Freeze',       self.set_freeze,                    False),
                     E(eT.CHCK, 'Prompt',       self.set_show_prompt,              False)])
        elem.append([E(eT.SEPR, '')])
        elem.append([E(eT.TEXT, 'PLAY     '),
                     E(eT.BTTN, 'Start',        self.sequencer.start_sequence),
                     E(eT.BTTN, 'Reset',        self.sequencer.reset)])
        elem.append([E(eT.SEPR, '')])
        elem.append([E(eT.TEXT, 'PROGRESS '),
                     E(eT.ITXT, 'Ani',          self.emptyFunction,                 'none', expand = False, size = (12,1)),
                     E(eT.PBAR, 'AniProgress',  None,                               0,      (0,1),   0.01)])
        elem.append([E(eT.TEXT, '         '),
                     E(eT.ITXT, 'State',        self.emptyFunction,                 'none', expand = False, size = (12,1)),
                     E(eT.PBAR, 'SeqProgress',  None,                               0,      (0,1),   0.01)])
        elem.append([E(eT.SEPR, '')])
        elem.append([E(eT.TEXT, 'TIMING    Dstry In '),
                     E(eT.SLDR, 'DstryInDrtn',  self.set_destroy_in_duration,       2,      (0.1,6),   0.1),
                     E(eT.CMBO, 'DstryInEase',  self.set_destroy_in_ease,           'Linear',   EasingNames, expand = False, size = (12,1))])
        elem.append([E(eT.TEXT, '          Dstry Out'),
                     E(eT.SLDR, 'DstryOutDrtn', self.set_destroy_out_duration,      2,      (0.1,6),   0.1),
                     E(eT.CMBO, 'DstryOutEase', self.set_destroy_out_ease,          'Linear',   EasingNames, expand = False, size = (12,1))])
        elem.append([E(eT.TEXT, '          Cam In   '),
                     E(eT.SLDR, 'CamDetDrtn',    self.set_cam_det_duration,           2,      (0.1,6),   0.1),
                     E(eT.CMBO, 'CamDetEase',    self.set_cam_det_ease,               'Linear',   EasingNames, expand = False, size = (12,1))])
        elem.append([E(eT.TEXT, '          Cam Out  '),
                     E(eT.SLDR, 'CamOutDrtn',   self.set_cam_rec_duration,          2,      (0.1,6),   0.1),
                     E(eT.CMBO, 'CamOutEase',   self.set_cam_rec_ease,              'Linear',   EasingNames, expand = False, size = (12,1))])
        elem.append([E(eT.TEXT, '          Flash In '),
                     E(eT.SLDR, 'FlashInDrtn',  self.set_flash_in_duration,         2,      (0.1,6),    0.1),
                     E(eT.CMBO, 'FlashInEase',  self.set_flash_in_ease,             'Linear',   EasingNames, expand = False, size = (12,1))])
        elem.append([E(eT.CHCK, 'Auto   Flash out',   self.set_flash_out_auto,      True),
                     E(eT.SLDR, 'FlashOutDrtn', self.set_flash_out_duration,        2,      (0.1,6),   0.1),
                     E(eT.CMBO, 'FlashOutEase', self.set_flash_out_ease,            'Linear',   EasingNames, expand = False, size = (12,1))])
        elem.append([E(eT.TEXT, '          Timeout  '),
                     E(eT.SLDR, 'Timeout',      self.set_timeout_duration,          2,      (0.1,6),    0.1),
                     E(eT.CMBO, 'TimeoutEase',  self.set_timeout_ease,              'Linear',   EasingNames, expand = False, size = (12,1))])
        elem.append([E(eT.SEPR, '')])
        elem.append([E(eT.TEXT, 'EFFECT   '),
                     E(eT.CHCK, 'Trail',        self.set_trail_active,              False),
                     E(eT.SLDR, 'Trail',        self.set_trail,                     0.995,  (0.98, 1),  0.001)])
        elem.append([E(eT.TEXT, '          Dstr Blur'),
                     E(eT.SLDR, 'DstrBlur',     self.set_dstr_blur,                 0.1,  (0, 1),     0.01),
                     E(eT.TEXT, 'Bloom'),
                     E(eT.SLDR, 'DstrBloom',    self.set_dstr_bloom,                0.6,  (0, 1),     0.1)])

        self.frame = Frame('SEQUENCE', elem, 540)

    def get_gui_frame(self):
        return self.frame

    def draw(self) -> None: # override
        super().draw()
        if not self.gui: return
        if not self.gui.isRunning(): return

        message: SeqenceMessage = self.sequencer.message
        self.gui.updateElement('State', message.sequence_type.name)
        self.gui.updateElement('SeqProgress', message.value_1)
        self.gui.updateElement('Ani', message.animation_type.name)
        self.gui.updateElement('AniProgress', message.progress)

        self.gui.updateElement('FlashOutDrtn', self.sequencer.flash_out_duration)

    def keyboard_callback(self, key, x, y) -> None: # override
        super().keyboard_callback(key, x, y)
        if key == b'f' or key == b'F':
            self.set_fullscreen(not self.fullscreen)
            if not self.gui or not self.gui.isRunning(): return
            self.gui.updateElement('Fullscreen', self.fullscreen)

        if key == b'g' or key == b'G':
            if not self.gui or not self.gui.isRunning(): return
            self.gui.bringToFront()

        if key == b'r' or key == b'R':
            self.reload_shaders()

        if key == b'\r':
            self.sequencer.start_sequence()

    def gui_set_draw_cam(self, value: bool) -> None:
        self.draw_camera = value

    def set_freeze(self, value:bool) -> None:
        self.freeze = value

    def set_show_prompt(self, value: bool) -> None:
        self.show_prompt = value


    def set_destroy_in_duration(self, value: float) -> None:
        self.sequencer.destroy_in_duration = value
    def set_destroy_in_ease(self, value: str) -> None:
        self.sequencer.destroy_in_ease = getEasing(value)

    def set_destroy_out_duration(self, value: float) -> None:
        self.sequencer.destroy_out_duration = value
    def set_destroy_out_ease(self, value: str) -> None:
        self.sequencer.destroy_out_ease = getEasing(value)

    def set_cam_det_duration(self, value: float) -> None:
        self.sequencer.cam_det_duration = value
    def set_cam_det_ease(self, value: str) -> None:
        self.sequencer.cam_det_ease = getEasing(value)

    def set_cam_rec_duration(self, value: float) -> None:
        self.sequencer.cam_rec_duration = value
    def set_cam_rec_ease(self, value: str) -> None:
        self.sequencer.cam_rec_ease = getEasing(value)

    def set_flash_in_duration(self, value: float) -> None:
        self.sequencer.flash_in_duration = value
    def set_flash_in_ease(self, value: str) -> None:
        self.sequencer.flash_in_ease = getEasing(value)

    def set_flash_out_duration(self, value: float) -> None:
        self.sequencer.flash_out_duration = value
    def set_flash_out_ease(self, value: str) -> None:
        self.sequencer.flash_out_ease = getEasing(value)
    def set_flash_out_auto(self, value: bool) -> None:
        self.sequencer.set_flash_out_auto(value)

    def set_timeout_duration(self, value: float) -> None:
        self.sequencer.timout_duration = value
    def set_timeout_ease(self, value: str) -> None:
        pass


    def set_trail_active(self, value:bool) -> None:
        self.trail_active = value
        if value:
            self.trail = self.trail_gui
        else:
            self.trail = 0

    def set_trail(self, value:float) -> None:
        self.trail_gui = value
        if self.trail_active:
            self.trail = self.trail_gui

    def set_dstr_blur(self, value: float) -> None:
        self.dstry_blur = value

    def set_dstr_bloom(self, value: float) -> None:
        self.dstry_bloom = value


    def emptyFunction(self, value) -> None:
        pass


def getEasing(type: str) -> Callable[[float], float]:
    if type == 'Sine In':
        return easing.easeInSine
    if type == 'Sine Out':
        return easing.easeOutSine
    if type == 'Sine In Out':
        return easing.easeInOutSine
    if type == 'Quad In':
        return easing.easeInQuad
    if type == 'Quad Out':
        return easing.easeOutQuad
    if type == 'Quad In Out':
        return easing.easeInOutQuad
    if type == 'Cubic In':
        return easing.easeInCubic
    if type == 'Cubic Out':
        return easing.easeOutCubic
    if type == 'Cubic In Out':
        return easing.easeInOutCubic
    if type == 'Quart In':
        return easing.easeInQuart
    if type == 'Quart Out':
        return easing.easeOutQuart
    if type == 'Quart In Out':
        return easing.easeInOutQuart
    if type == 'Quint In':
        return easing.easeInQuint
    if type == 'Quint Out':
        return easing.easeOutQuint
    if type == 'Quint In Out':
        return easing.easeInOutQuint

    return easing.linear
