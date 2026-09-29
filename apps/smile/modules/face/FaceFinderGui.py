from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame
from modules.face.FaceFinder import FaceFinder, FaceMessage

class FaceFinderGui(FaceFinder):
    def __init__(self, gui: Gui | None, composition_width: int, composition_height: int) -> None:
        super().__init__(composition_width, composition_height)
        self.gui: Gui | None = gui

        elem: list = []
        elem.append([E(eT.TEXT, 'FILTER   '),
                     E(eT.TEXT, 'Size'),
                     E(eT.SLDR, 'SizeFilter',       self.set_size_filter,       0.1,   (0, 0.3),        .01),
                     E(eT.TEXT, 'Conf'),
                     E(eT.SLDR, 'FaceConfidence',   self.set_conf_filter,       0.7,   (0.5, 1.0),      .01)])
        elem.append([E(eT.TEXT, 'TRIGGER  '),
                     E(eT.TEXT, 'Frames'),
                     E(eT.SLDR, 'FaceFrames',       self.set_stable_frames,     3,     (1, 10),         1)])
        elem.append([E(eT.TEXT, 'PORTRAIT  '),
                     E(eT.TEXT, 'Zoom'),
                     E(eT.SLDR, 'p_zoom',           self.set_zoom,              1,     (1, 5),        0.1),
                     E(eT.TEXT, 'y offset'),
                     E(eT.SLDR, 'p_offset',         self.set_offset,            0,     (-.3, .3),        0.01)])
        elem.append([E(eT.TEXT, 'RESULT   '),
                     E(eT.MLTL, 'faceresult',       None,       'TEST',     resolution=2, expand = False)])

        self.frame = Frame('FACE FINDER', elem, 150)

    def get_gui_frame(self):
          return self.frame

    def message_callback(self, message: FaceMessage) -> None: # override
        super().message_callback(message)
        face_txt: str = 'size: ' + str(message.size) + ', score: ' + str(message.score) + ', stable: ' + str(message.stable)
        if self.gui:
            self.gui.updateElement('faceresult', face_txt)

    def error_callback(self, error: ValueError) -> None: # override
        if self.gui:
            self.gui.updateElement('faceresult', 'error: ' + str(error))

    def set_size_filter(self, value: float) -> None:
        self.size_filter = value

    def set_conf_filter(self, value: float) -> None:
        self.confidence_filter = value

    def set_stable_frames(self, value: float) -> None:
        self.stable_frames = int(value)

    def set_zoom(self, value: float) -> None:
        self.zoom = value

    def set_offset(self, value: float) -> None:
        self.offset = value
