from modules.cam.depthplayer.SyncPlayer import *
from modules.gui.PyReallySimpleGui import Gui, eType as eT
from modules.gui.PyReallySimpleGui import Element as E, Frame

class SyncPlayerGui(SyncPlayer):

    def __init__(self, gui: Gui | None, settings: Settings) -> None:
        self.gui: Gui | None = gui
        super().__init__(settings)

        folders: list[str] = self.get_folder_names()
        f: str = folders[0] if len(folders) > 0 else ''

        elem: list = []
        elem.append([E(eT.TEXT, 'Folders '),
                     E(eT.CMBO, 'Folders',  self.set_folder, f, folders),
                     E(eT.BTTN, 'P_Start',  self.set_start),
                     E(eT.BTTN, 'P_Stop',   self.set_stop)])
        elem.append([E(eT.TEXT, 'Chunks  '),
                     E(eT.ITXT, 'C_Chunk',  self.nothing, expand = False),
                     E(eT.TEXT, 'of'),
                     E(eT.ITXT, 'M_Chunk',  self.nothing, expand = False),
                     E(eT.TEXT, 'Range'),
                     E(eT.ITXT, 'C_R_0',    self.set_chunk_range_0, '0', expand = False),
                     E(eT.TEXT, 'to'),
                     E(eT.ITXT, 'C_R_1',    self.set_chunk_range_1, '0', expand = False)])

        self.frame = Frame('CAMERA COLOR', elem, 80)

    def get_gui_frame(self):
        return self.frame

    def gui_check(self) -> None:
        return

    def set_folder(self, folder: str) -> None:
        if self.gui is None:
            return

        folders: list[str] = self.get_folder_names()
        if not folders:
            self.gui.updateElement('Folders', 'No folders available')
            return

        if not folder in folders:
            folder = folders[0]
            self.gui.updateElement('Folders', folder)

        self.play(True, folder)
        num_chunks: int = self.get_num_folder_chunks(folder)

        self.gui.updateElement('M_Chunk', num_chunks)
        self.gui.updateElement('C_R_0', str(0))
        self.gui.updateElement('C_R_1', str(num_chunks))

    def set_start(self) -> None:
        if self.gui is None:
            return
        f: str = self.gui.getStringValue('Folders')

        self.play(True, f)

    def set_stop(self) -> None:
        self.play(False, '')

    def set_chunk_range_0(self, R0_str: str) -> None:
        if self.gui is None:
            return
        if R0_str == '':
            return
        R0: int = 0
        if R0_str.isdigit():
            R0 = int(R0_str)
        else:
            self.gui.updateElement('C_R_0', R0)
        if R0 > self.get_num_chunks():
            R0 = self.get_num_chunks()
            self.gui.updateElement('C_R_0', R0)
        R1_str: str = self.gui.getStringValue('C_R_1')
        R1: int = self.get_num_chunks()
        if R1_str.isdigit():
            R1: int = int(R1_str)
        if R1 < R0:
            R1 = R0
            self.gui.updateElement('C_R_1', R1)
        self.set_chunk_range(R0, R1)
        self.set_start()

    def set_chunk_range_1(self, R1_str: str) -> None:
        if self.gui is None:
            return
        if R1_str == '':
            return
        R1: int = self.get_num_chunks()
        if R1_str.isdigit():
            R1 = int(R1_str)
        else:
            self.gui.updateElement('C_R_1', R1)
        if R1 > self.get_num_chunks():
            R1 = self.get_num_chunks()
            self.gui.updateElement('C_R_1', R1)
        R0_str: str = self.gui.getStringValue('C_R_0')
        R0: int = 0
        if R0_str.isdigit():
            R0: int = int(R0_str)
        if R1 < R0:
            R0 = R1
            self.gui.updateElement('C_R_0', R0)
        self.set_chunk_range(R0, R1)

    def update_gui(self) -> None: #override
        if self.gui is None:
            return
        self.gui.updateElement('C_Chunk', self.get_current_chunk())

    def nothing(self, int) -> None:
        pass

