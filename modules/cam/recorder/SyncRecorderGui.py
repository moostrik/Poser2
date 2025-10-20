from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame
from modules.cam.recorder.SyncRecorder import SyncRecorder
from modules.cam.depthcam.Definitions import FrameType
from modules.utils.HotReloadMethods import HotReloadMethods
import time
from threading import Thread

from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.dispatcher import Dispatcher

from modules.Settings import Settings
class SyncRecorderGui(SyncRecorder):
    def __init__(self, gui: Gui | None, settings: Settings) -> None:
        self.gui: Gui | None = gui
        super().__init__(settings)

        self.osc_client = SimpleUDPClient("10.0.0.148", 8600)

        self.osc_receive: Dispatcher = Dispatcher()
        self.osc_receive.set_default_handler(self.receive_messages)
        self.server = ThreadingOSCUDPServer(('0.0.0.0', 8601), self.osc_receive)
        self.server_thread: Thread = Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        self.recording: bool = False

        elem: list = []
        elem.append([E(eT.BTTN, 'Start Recording', self.gui_start_recording_from_gui, False),
                     E(eT.BTTN, 'Stop Recording', self.gui_stop_recording, False),
                     E(eT.CHCK, 'Rec', None, False),
                     E(eT.ITXT, 'Group ID', self.receive_text_for_no_reason, 'no_id')])

        self._frame = Frame('RECORDER', elem, 60)
        hot_reload = HotReloadMethods(self.__class__, True, True)

    def get_gui_frame(self):
        return self._frame

    def gui_check(self) -> None:
        if self.gui is not None:
            self.gui.updateElement('Rec', False)

    def gui_start_recording_from_gui(self) -> None:
        if self.recording:
            return
        self.recording = True
        self.gui.updateElement('Group ID', 'no_id')
        self.gui.updateElement('Rec', True)

        self.set_group_id('no_id')
        self.record(True)

    def gui_start_recording(self) -> None:
        if self.recording:
            return
        self.recording = True
        self.gui.updateElement('Rec', True)

        self.record(True)

    def gui_stop_recording(self) -> None:
        if not self.recording:
            return
        self.recording = False
        self.gui.updateElement('Rec', False)

        self.record(False)

    def receive_messages(self, adress, *args):
        print(adress, args)

        if adress == '/group/id':
            self.set_group_id(args[0])
            self.gui.updateElement('Group ID', args[0])
        if adress == '/start/recording':
            self.gui_start_recording()
        if adress == '/stop/recording':
            self.gui_stop_recording()

        self.osc_client.send_message(adress, args)

    def receive_text_for_no_reason(self, somestring):
        pass