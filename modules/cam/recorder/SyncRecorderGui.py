from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame
from modules.cam.recorder.SyncRecorder import SyncRecorder
from modules.cam.depthcam.Definitions import FrameType
from modules.utils.HotReloadMethods import HotReloadMethods
import threading
import time

from pythonosc.udp_client import SimpleUDPClient

from modules.Settings import Settings
class SyncRecorderGui(SyncRecorder):
    def __init__(self, gui: Gui | None, settings: Settings) -> None:
        self.gui: Gui | None = gui
        super().__init__(settings)

        self.osc_clients = [
            SimpleUDPClient("172.119.0.24", port) for port in range(8600, 8610)
        ]

        elem: list = []
        elem.append([E(eT.CHCK, 'Rec',    self.gui_record, False),
                     E(eT.BTTN, 'End', self.gui_marker_1, False),
                     E(eT.BTTN, 'Baseline', self.gui_marker_2, False)])

        self.rec_signal: bool = False
        self.rec_signal_time: float = 0.0
        self.marker_signal_1: bool = False
        self.marker_signal_1_time: float = 0.0
        self.marker_signal_2: bool = False
        self.marker_signal_2_time: float = 0.0

        self.osc_loop: threading.Thread = threading.Thread(target=self.send_osc_loop, daemon=True)
        self.osc_loop.start()

        self._frame = Frame('RECORDER', elem, 60)
        hot_reload = HotReloadMethods(self.__class__, True, True)

    def get_gui_frame(self):
        return self._frame

    def gui_check(self) -> None:
        if self.gui is not None:
            self.gui.updateElement('Rec', False)


    def gui_record(self, rec: bool) -> None:
        self.record(rec)
        print(f"Sending record command: {rec}")
        if rec == True:
            self.rec_signal = True
            self.rec_signal_time = time.time()
        # self.osc_client.send_message("/HDT/record", int(rec))

    def gui_marker_1(self) -> None:
        print("Sending signal to HDT")
        self.marker_signal_1 = True
        self.marker_signal_1_time = time.time()
        # self.osc_client.send_message("/HDT/signal", 1)

    def gui_marker_2(self) -> None:
        print("Sending signal to HDT")
        self.marker_signal_2 = True
        self.marker_signal_2_time = time.time()
        # self.osc_client.send_message("/HDT/signal", 2)

    def send_osc_loop(self):
        while True:
            if self.rec_signal and (time.time() - self.rec_signal_time > 0.5):
                self.rec_signal = False
            for i, client in enumerate(self.osc_clients):
                client.send_message(f"/HDT/record", int(self.rec_signal))

            if self.marker_signal_1 and (time.time() - self.marker_signal_1_time > 0.5):
                self.marker_signal_1 = False
            for i, client in enumerate(self.osc_clients):
                client.send_message(f"/HDT/end", int(self.marker_signal_1))

            if self.marker_signal_2 and (time.time() - self.marker_signal_2_time > 0.5):
                self.marker_signal_2 = False
            for i, client in enumerate(self.osc_clients):
                client.send_message(f"/HDT/baseline", int(self.marker_signal_2))

            time.sleep(0.1)