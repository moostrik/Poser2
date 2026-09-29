from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame
from modules.artnet.Sender import Sender

class SenderGui(Sender):
    def __init__(self, gui: Gui, settings_file: str, ip: str = "127.0.0.1", universe: int = 1, fps: float = 30.0) -> None:
        super().__init__(settings_file, ip, universe, fps)
        self.gui: Gui = gui

        elem: list = []
        elem.append([E(eT.CHCK, 'Artnet_Active',   self.set_sending,   True),
                     E(eT.ITXT, 'Artnet_IP',       self.set_ip,        self.ip),
                     E(eT.ITXT, 'Artnet_Uni',      self.set_universe,  self.universe),
                     E(eT.ITXT, 'Artnet_FPS',      self.set_fps,       self.fps)])

        self.frame = Frame('ArtNet', elem, 50)

    def get_gui_frame(self):
        return self.frame
