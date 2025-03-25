from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame
from modules.cam.recorder.SyncRecorder import SyncRecorder
from modules.cam.DepthAi.Definitions import FrameType, FrameTypeNames

class SyncRecorderGui(SyncRecorder):
    def __init__(self, gui: Gui | None, output_path: str, types: list[FrameType], chunk_duration: float) -> None:
        self.gui: Gui | None = gui
        super().__init__(output_path, types, chunk_duration)

        elem: list = []
        elem.append([E(eT.CHCK, 'Rec',    self.check_recording, False),
                     E(eT.TEXT, 'Rec_Text',    None)])


        self._frame = Frame('RECORDER', elem, 60)

    def get_gui_frame(self):
        return self._frame

    def check_recording(self, value) -> None:
        if value and not self.recording:
            self.start_recording()
        if not value and self.recording:
            self.stop_recording()