from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame
from modules.osc.Sender import Sender

class SenderGui(Sender):
    def __init__(self, gui: Gui, ip: str = "127.0.0.1", port: int | str = 8000, fps: float = 30.0) -> None:
        super().__init__(ip, port, fps)
        self.gui: Gui = gui

        elem: list = []
        elem.append([E(eT.CHCK, 'OSC_Active',   self.set_sending,    True),
                     E(eT.ITXT, 'OSC_IP',       self.set_ip,        self.ip),
                     E(eT.ITXT, 'OSC_Port',     self.set_port,      self.port),
                     E(eT.ITXT, 'OSC_FPS',      self.set_fps,       self.fps)])

        self.frame = Frame('OSC', elem, 50)

    def get_gui_frame(self):
        return self.frame
