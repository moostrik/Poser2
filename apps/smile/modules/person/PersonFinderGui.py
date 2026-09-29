from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame
from modules.person.PersonFinder import PersonFinder, PersonMessage

class PersonFinderGui(PersonFinder):
    def __init__(self, gui: Gui | None) -> None:
        super().__init__()
        self.gui: Gui | None = gui

        elem: list = []
        elem.append([E(eT.TEXT, 'FILTER   '),
                     E(eT.TEXT, 'Confid'),
                     E(eT.SLDR, 'PersonConfidence',     self.set_confidence,    0.85,   (0.5, 1.0),     .01),
                     E(eT.TEXT, 'Size'),
                     E(eT.SLDR, 'PersonSizeFilter',     self.set_size_filter,   0.1,    (0, 0.3),       .01)])
        elem.append([E(eT.TEXT, '         '),
                     E(eT.TEXT, 'Bottom'),
                     E(eT.SLDR, 'PersonBottomFilter',   self.set_bottom_filter, 0.99,   (0.5, 1.0),    .01),
                     E(eT.TEXT, 'Edge'),
                     E(eT.SLDR, 'PersonEdgeFilter',     self.set_edge_filter,   0.1,    (0, 0.3),       .01)])

        self.frame = Frame('PERSON FINDER', elem, 100)

    def get_gui_frame(self):
          return self.frame
