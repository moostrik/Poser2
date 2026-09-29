from modules.face.FaceFinderGui import FaceFinderGui as FF, FaceMessage
from modules.person.PersonFinderGui import PersonFinderGui as PF, PersonMessage
from modules.gui.PyReallySimpleGui import Gui
from modules.render.RenderGui import RenderGui as Render
from modules.render.Sequencer import SeqenceMessage, SequenceType, ContinueType, AnimationType, AnimationTypeNames
from modules.sd.WebUiGui import WebUiGui as SD
from modules.sd.WebUi import WebUiMessage, WebUiState
from modules.osc.SenderGui import SenderGui as Osc
from modules.audio.DBGui import DBGui as AudioDB
from modules.save.ImgFolderGui import ImgFolderGui
from modules.artnet.SenderGui import SenderGui as Artnet
from enum import Enum
import numpy as np

class CamType(Enum):
    VIMBA   = 2
    WEB     = 3
    IMAGE   = 4

class Forgettable():
    def __init__(self, path: str, width: int, height: int, portraitMode: bool = False, guiTop: bool = False, camType: CamType = CamType.VIMBA) -> None:
        self.path: str = path
        self.width: int = width
        self.height: int = height
        self.portraitMode: bool = portraitMode
        if self.portraitMode:
            self.width: int = height
            self.height: int = width

        self.guiTop: bool = guiTop

        self.gui: Gui = Gui('AIIA', path + '/settings/', 'default')
        self.render: Render = Render(self.gui, self.width, self.height , self.width, self.height, 'aiia', fullscreen=False, v_sync=True, stretch=False)
        self.person: PF = PF(self.gui)
        self.face: FF = FF(self.gui, self.width, self.height)
        self.diffuse = SD(self.gui, path + '/stable-diffusion-webui/')
        self.diffuse.set_portrait_mode(self.portraitMode)

        if camType == CamType.VIMBA:
            from modules.cam.VimbaCamGui import VimbaCamGui as VimbaCam
            self.camera = VimbaCam(self.gui, (self.width, self.height), False)
        elif camType == CamType.WEB:
            from modules.cam.WebCamGui import WebCamGui as WebCam
            self.camera = WebCam(self.gui, (self.width, self.height), False)
        else:
            from modules.cam.ImageGui import ImageGui as ImageCam
            self.camera = ImageCam(self.gui, (self.width, self.height), False)
        self.camera.setRotate90(self.portraitMode)

        self.audio: AudioDB = AudioDB(self.gui, 'files/music_map.json')
        self.osc: Osc = Osc(self.gui, "127.0.0.1", 4114, 30.0)
        self.save: ImgFolderGui = ImgFolderGui(self.gui, min_files=16)
        self.artnet = Artnet(self.gui, 'files/artnet.json')

        self._running: bool = False

        self.laststate: str = ''

        self.face_messages: list[FaceMessage] = []
        self.person_messages: list[PersonMessage] = []


    def start(self) -> None:
        self.camera.setColorCallback(self.render.set_cam_image)
        self.camera.open()
        self.camera.startCapture()

        self.diffuse.add_message_callback(self.render.set_diffusion_message)
        self.diffuse.add_message_callback(self.diffusion_callback)
        self.diffuse.start()

        self.person.add_message_callback(self.person_callback)
        self.person.start()

        self.face.add_message_callback(self.face_callback)
        self.face.start()

        self.render.exit_callback = self.stop
        self.render.sequencer.add_message_callback(self.sequence_callback)
        self.render.set_vote_visual_callback(self.vote_visual_callback)
        self.render.set_vote_audio_callback(self.vote_audio_callback)
        self.render.set_remove_audio_callback(self.remove_audio_callback)
        self.render.start()

        self.osc.start()
        self.artnet.start()

        self.gui.exit_callback = self.stop

        if self.width < self.height:
            self.gui.addFrame([self.diffuse.get_frame_diff(), self.diffuse.get_frame_prompt(), self.render.get_gui_frame()])
            self.gui.addFrame([self.camera.get_gui_frame(), self.face.get_gui_frame(), self.person.get_gui_frame()])
            self.gui.addFrame([self.audio.get_gui_frame(), self.osc.get_gui_frame(), self.artnet.get_gui_frame()])
        else:
            self.gui.addFrame([self.diffuse.get_frame_diff(), self.diffuse.get_frame_prompt()])
            self.gui.addFrame([self.camera.get_gui_frame(), self.render.get_gui_frame()])
            self.gui.addFrame([self.face.get_gui_frame(), self.save.get_gui_frame()])

        self.gui.start()

        if not self.guiTop:
            self.gui.sendToBack()

        self._running = True

    def stop(self) -> None:
        self.artnet.stop()
        self.osc.stop()

        self.camera.clearColorCallbacks()
        self.camera.close()

        self.diffuse.clear_message_callbacks()
        self.diffuse.stop()

        self.person.clear_message_callbacks()
        self.person.stop()

        self.face.clear_message_callbacks()
        self.face.stop()

        self.render.exit_callback = None
        self.render.sequencer.clear_message_callbacks()
        self.render.stop()

        self.gui.exit_callback = None
        self.gui.stop()


        self._running = False

    def sequence_callback(self, message: SeqenceMessage) -> None:
        seq: SequenceType = message.sequence_type
        ani: AnimationType = message.animation_type
        continue_type: ContinueType = self.render.sequencer.get_continue_type()
        state: str = AnimationTypeNames[message.animation_type]

        cam_image: np.ndarray= self.camera.getColorImg()
        artnet_progress: float = message.progress
        if seq == SequenceType.TIMEOUT:
            artnet_progress = 1.0
        if seq == SequenceType.CAMERA_DETECT:
            artnet_progress = 0.0
        self.artnet.set_progress(artnet_progress, ani)

        if self.osc.sending:
            progress: float = message.progress
            audio_id: str = self.audio.get_audio_id()
            self.osc.send('/music_id', audio_id)
            self.osc.send('/state', state)
            self.osc.send('/progress', progress)

        if seq == SequenceType.CAMERA_SHOW:
            self.face_messages = []
            self.person_messages = []
            self.face.reset()

        if seq >= SequenceType.CAMERA_SHOW and seq <= SequenceType.CAMERA_REC:
            if continue_type == ContinueType.PERSON:
                self.person.set_image(cam_image)
            if continue_type == ContinueType.FACE:
                self.face.set_image(cam_image)

        if seq == SequenceType.FLASH:
            if message.initial == True:
                movement_id: str = self.diffuse.prompter.get_keyword_next_id('smile')
                self.audio.update(movement_id)

        if seq == SequenceType.CAPTURE:
            cap_image: np.ndarray = cam_image

            if continue_type == ContinueType.PERSON:
                cap_image = self.get_last_person(cam_image)

            if continue_type == ContinueType.FACE:
                portrait: FaceMessage | None = self.get_best_face()
                if portrait is not None:
                    cap_image = portrait.image

            self.render.set_cap_image(cap_image)
            self.diffuse.set_image(cap_image)
            self.save.new_sequence()
            self.save.save_image(cap_image)


        if seq == SequenceType.TIMEOUT:
            if message.initial == True:
                self.save.end_sequence()

        if seq == SequenceType.RESET:
            self.diffuse.set_interrupt_gui()

    def get_last_person(self, image: np.ndarray) -> np.ndarray:
        if self.person_messages:
            if self.person_messages[-1].image is not None:
                # if last person message has image, return the current (and latest) image
                return image
            else:
                # return last persons image that was found
                for p in reversed(self.person_messages):
                    if p.image is not None:
                        return p.image
        # if no person was found, return the current image
        print('warning, persons not set')
        return image

    def get_best_face(self, window: float = 1.0) -> FaceMessage | None:
        # highest scoring face of the last second, so a blink or a turned head at the last frame is not used
        if not self.face_messages:
            print('warning, faces not set')
            return None
        latest: float = self.face_messages[-1].time
        recent: list[FaceMessage] = [f for f in self.face_messages if f.time >= latest - window]
        return max(recent, key=lambda f: f.score)

    def person_callback(self, message: PersonMessage) -> None:
        self.person_messages.append(message)
        if message.found:
            self.render.sequencer.set_person_found()

    def face_callback(self, message: FaceMessage) -> None:
        self.face_messages.append(message)
        if message.stable:
            self.render.sequencer.set_face_found()

    def diffusion_callback(self, message: WebUiMessage) -> None:
        if message.state == WebUiState.DIFFUSION or message.state == WebUiState.DIFF_LOOP:
            if message.image is not None:
                self.save.save_image(message.image)

    def vote_visual_callback(self) -> None:
        self.diffuse.vote()

    def vote_audio_callback(self) -> None:
        self.audio.vote()

    def remove_audio_callback(self) -> None:
        self.audio.reject()

    def isRunning(self) -> bool :
        return self._running
