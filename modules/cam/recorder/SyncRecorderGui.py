from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame
from modules.cam.recorder.SyncRecorder import SyncRecorder
from modules.cam.depthcam.Definitions import FrameType


from pythonosc.udp_client import SimpleUDPClient

from modules.Settings import Settings
class SyncRecorderGui(SyncRecorder):
    def __init__(self, gui: Gui | None, settings: Settings) -> None:
        self.gui: Gui | None = gui
        super().__init__(settings)

        self.osc_client = SimpleUDPClient("127.0.0.1", 9000)  # Change IP/port as needed

        elem: list = []
        elem.append([E(eT.CHCK, 'Rec',    self.record, False),
                     E(eT.BTTN, 'Signal', self.gui_signal, False),])

        self._frame = Frame('RECORDER', elem, 60)

    def get_gui_frame(self):
        return self._frame

    def gui_check(self) -> None:
        if self.gui is not None:
            self.gui.updateElement('Rec', False)


    def gui_record(self, rec: bool) -> None:
        self.record(rec)
        self.osc_client.send_message("/HDT/record", int(rec))

    def gui_signal(self) -> None:
        self.osc_client.send_message("/HDT/signal", 1)
