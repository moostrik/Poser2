from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame
from modules.cam.recorder.SyncRecorder import SyncRecorder, EncoderType
from modules.cam.DepthAi.Definitions import FrameType, FrameTypeNames

class SyncRecorderGui(SyncRecorder):
    def __init__(self, gui: Gui | None, output_path: str, num_cams: int, types: list[FrameType], chunk_duration: float, encoder: EncoderType) -> None:
        self.gui: Gui | None = gui
        super().__init__(output_path, num_cams, types, chunk_duration, encoder)

        elem: list = []
        elem.append([E(eT.CHCK, 'Rec',    self.check_recording, False),
                     E(eT.TEXT, 'Rec_Text',    None)])


        self._frame = Frame('RECORDER', elem, 60)

    def get_gui_frame(self):
        return self._frame

    def check_recording(self, value) -> None:
        if value:
            self.start_recording()
        if not value:
            self.stop_recording()