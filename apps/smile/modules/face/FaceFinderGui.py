from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame
from modules.face.FaceFinder import FaceFinder, FaceMessage
import json
from modules.sd.WebUi import WebUi

class FaceFinderGui(FaceFinder):
    def __init__(self, gui: Gui | None, composition_width: int, composition_height: int, web_ui: WebUi) -> None:
        super().__init__(composition_width, composition_height, web_ui)
        self.gui: Gui | None = gui

        elem: list = []
        elem.append([E(eT.TEXT, 'FILTER   '),
                     E(eT.TEXT, 'Size'),
                     E(eT.SLDR, 'SizeFilter',       self.set_size_filter,       0,     (0, 0.3),        .01),
                     E(eT.TEXT, 'Conf'),
                     E(eT.SLDR, 'ConfFilter',       self.set_conf_filter,       5,     (0, 10),         .5)])
        elem.append([E(eT.TEXT, 'PORTRAIT  '),
                     E(eT.TEXT, 'Zoom'),
                     E(eT.SLDR, 'p_zoom',           self.set_zoom,              1,     (1, 5),        0.1),
                     E(eT.TEXT, 'y offset'),
                     E(eT.SLDR, 'p_offset',         self.set_offset,            0,     (-.3, .3),        0.01)])
        elem.append([E(eT.TEXT, 'PROMPT  '),
                     E(eT.CHCK, 'emotion',          self.set_prompt_emotion,    True),
                     E(eT.CHCK, 'age',              self.set_prompt_age,        True),
                     E(eT.CHCK, 'race',             self.set_prompt_race,       True),
                     E(eT.CHCK, 'gender',           self.set_prompt_gender,     True)])
        elem.append([E(eT.TEXT, 'RESULT   '),
                     E(eT.MLTL, 'faceresult',       None,       'TEST',     resolution=2, expand = False)])

        self.frame = Frame('FACE FINDER', elem, 150)

    def get_gui_frame(self):
          return self.frame



    def message_callback(self, message: FaceMessage) -> None: # override
        super().message_callback(message)
        face_txt: str = ''
        face_txt += 'size: ' + json.dumps(message.size)
        face_txt += ', score: ' + json.dumps(message.score)
        face_txt += '\n' 'prompt: ' + json.dumps(message.prompt)

        if self.gui:
            self.gui.updateElement('faceresult', face_txt)

    def error_callback(self, error: ValueError) -> None: # override
        if self.gui:
            self.gui.updateElement('faceresult', 'error: ' + str(error))

    def set_size_filter(self, value: float) -> None:
        self.size_filter = value

    def set_edge_filter(self, value: float) -> None:
        self.edge_filter = value

    def set_conf_filter(self, value: float) -> None:
        self.confidence_filter = value

    def set_zoom(self, value: float) -> None:
        self.zoom = value

    def set_offset(self, value: float) -> None:
        self.offset = value

    def set_rect_select(self, value: int) -> None:
        self.rect_select = value

    def set_prompt_emotion(self, value: bool) -> None:
        self.prompt_emotion = value

    def set_prompt_age(self, value: bool) -> None:
        self.prompt_age = value

    def set_prompt_race(self, value: bool) -> None:
        self.prompt_race = value

    def set_prompt_gender(self, value: bool) -> None:
        self.prompt_gender = value

