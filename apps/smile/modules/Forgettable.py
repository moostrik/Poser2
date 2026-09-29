from modules.face.FaceFinder import FaceFinder, FaceMessage
from modules.render.Render import Render
from modules.render.Sequencer import SeqenceMessage, SequenceType, AnimationType, AnimationTypeNames
from modules.sd.WebUi import WebUi as SD, WebUiMessage, WebUiState
from modules.osc.Sender import Sender as Osc
from modules.audio.DB import DB as AudioDB
from modules.save.ImgFolder import ImgFolder
from modules.artnet.Sender import Sender as Artnet
from modules.settings import presets, NiceServer
from modules.smile_settings import Settings
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)

class CamType(Enum):
    VIMBA   = 2
    WEB     = 3
    IMAGE   = 4

class Forgettable():
    def __init__(self, path: str, width: int, height: int, portraitMode: bool = False, camType: CamType = CamType.VIMBA) -> None:
        self.path: str = path
        self.width: int = width
        self.height: int = height
        self.portraitMode: bool = portraitMode
        if self.portraitMode:
            self.width: int = height
            self.height: int = width

        # SETTINGS
        presets.set_app('smile')
        self.settings = Settings()
        self.settings.server.title = 'SMILE'
        self.settings.server.port = 667
        preset_file = presets.startup_path()
        if not presets.load(self.settings, preset_file):
            logger.warning('No preset at %s, using defaults', preset_file)
        self.settings.initialize()
        self.settings_server = NiceServer(self.settings, self.settings.server, on_exit=self.stop)

        self.render: Render = Render(self.settings.render, self.settings.sequence, self.width, self.height , self.width, self.height, 'aiia', v_sync=True, stretch=False)
        self.face: FaceFinder = FaceFinder(self.settings.face, self.width, self.height)
        self.diffuse = SD(path + '/stable-diffusion-webui/')
        self.diffuse.set_portrait_mode(self.portraitMode)

        if camType == CamType.VIMBA:
            from modules.cam.VimbaCam import VimbaCam
            self.camera = VimbaCam(self.settings.camera, (self.width, self.height))
        elif camType == CamType.WEB:
            from modules.cam.WebCam import WebCam
            self.camera = WebCam(self.settings.camera, (self.width, self.height))
        else:
            from modules.cam.Image import Image as ImageCam
            self.camera = ImageCam(self.settings.camera, (self.width, self.height))
        self.camera.setRotate90(self.portraitMode)

        self.audio: AudioDB = AudioDB(self.settings.audio, 'files/music_map.json')
        self.osc: Osc = Osc(self.settings.osc)
        self.save: ImgFolder = ImgFolder(self.settings.save)
        self.artnet = Artnet(self.settings.artnet, 'files/artnet.json')

        self._running: bool = False
        self._stopping: bool = False

        self.laststate: str = ''

        self.face_messages: list[FaceMessage] = []


    def start(self) -> None:
        self.camera.setColorCallback(self.render.set_cam_image)
        self.camera.open()
        self.camera.startCapture()

        self.diffuse.add_message_callback(self.render.set_diffusion_message)
        self.diffuse.add_message_callback(self.diffusion_callback)
        self.diffuse.start()

        self.face.add_message_callback(self.face_callback)
        self.face.start()

        self.render.exit_callback = self.stop
        self.render.sequencer.add_message_callback(self.sequence_callback)
        self.render.start()

        self.osc.start()
        self.artnet.start()

        self.settings_server.start()

        self._running = True

    def stop(self) -> None:
        # called from the render window (Esc), the settings panel (Exit) and Ctrl+C; run once
        if self._stopping:
            return
        self._stopping = True

        self.artnet.stop()
        self.osc.stop()

        self.camera.clearColorCallbacks()
        self.camera.close()

        self.diffuse.clear_message_callbacks()
        self.diffuse.stop()

        self.face.clear_message_callbacks()
        self.face.stop()

        self.save.stop()

        self.render.exit_callback = None
        self.render.sequencer.clear_message_callbacks()
        self.render.stop()

        self.settings_server.stop()

        self._running = False

    def sequence_callback(self, message: SeqenceMessage) -> None:
        seq: SequenceType = message.sequence_type
        ani: AnimationType = message.animation_type
        state: str = AnimationTypeNames[message.animation_type]

        cam_image: np.ndarray= self.camera.getColorImg()
        artnet_progress: float = message.progress
        if seq == SequenceType.TIMEOUT:
            artnet_progress = 1.0
        if seq == SequenceType.CAMERA_DETECT:
            artnet_progress = 0.0
        self.artnet.set_progress(artnet_progress, ani)

        if self.settings.osc.active:
            progress: float = message.progress
            audio_id: str = self.audio.get_audio_id()
            self.osc.send('/music_id', audio_id)
            self.osc.send('/state', state)
            self.osc.send('/progress', progress)

        if seq == SequenceType.CAMERA_SHOW and message.initial:
            self.face_messages = []
            self.face.reset()

        if seq >= SequenceType.CAMERA_SHOW and seq <= SequenceType.CAMERA_REC:
            self.face.set_image(cam_image)

        if seq == SequenceType.FLASH:
            if message.initial == True:
                movement_id: str = self.diffuse.prompter.get_keyword_next_id('smile')
                self.audio.update(movement_id)

        if seq == SequenceType.CAPTURE:
            cap_image: np.ndarray = cam_image
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
            self.diffuse.interrupt()

    def get_best_face(self, window: float = 1.0) -> FaceMessage | None:
        # highest scoring face of the last second, so a blink or a turned head at the last frame is not used
        if not self.face_messages:
            print('warning, faces not set')
            return None
        latest: float = self.face_messages[-1].time
        recent: list[FaceMessage] = [f for f in self.face_messages if f.time >= latest - window]
        return max(recent, key=lambda f: f.score)

    def face_callback(self, message: FaceMessage) -> None:
        self.face_messages.append(message)
        if message.stable:
            self.render.sequencer.set_face_found()

    def diffusion_callback(self, message: WebUiMessage) -> None:
        if message.state == WebUiState.DIFFUSION or message.state == WebUiState.DIFF_LOOP:
            if message.image is not None:
                self.save.save_image(message.image)

    def isRunning(self) -> bool :
        return self._running
