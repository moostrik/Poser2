from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame as Frame
from modules.cam.recorder.SyncRecorder import SyncRecorder
from modules.cam.recorder.FFmpegRecorder import EncoderType
from modules.cam.depthcam.Definitions import FrameType

class SyncRecorderGui(SyncRecorder):
    def __init__(self, gui: Gui | None, output_path: str, temp_path: str, num_cams: int, types: list[FrameType], chunk_duration: float, encoder: EncoderType) -> None:
        self.gui: Gui | None = gui
        super().__init__(output_path, temp_path, num_cams, types, chunk_duration, encoder)

        elem: list = []
        elem.append([E(eT.CHCK, 'Rec',    self.record, False),
                     E(eT.TEXT, 'Rec_Text',    None)])

        self._frame = Frame('RECORDER', elem, 60)

    def get_gui_frame(self):
        return self._frame

    def gui_check(self) -> None:
        if self.gui is not None:
            self.gui.updateElement('Rec', False)