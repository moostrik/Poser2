from modules.cam.Camera import Camera
from modules.face.FaceFinder import FaceFinder, FaceFrame
from modules.render.SmileRender import SmileRender
from modules.render.Sequencer import Sequencer, SeqenceMessage, SequenceType, AnimationType, AnimationTypeNames
from modules.comfy.Generator import Generator
from modules.render.GrinPlayer import GrinPlayer
from modules.comfy.messages import GenMessage, GenState
from modules.osc.Sender import Sender as Osc
from modules.audio.DB import DB as AudioDB
from modules.audio.ScreenTest import ScreenTest
from modules.save.ClipSaver import ClipSaver
from modules.artnet.Sender import Sender as Artnet
from modules.settings import presets, NiceServer
from modules.gl import FullscreenMode
from modules.main_settings import Settings
from modules.log_config import crash_summary
from enum import Enum
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)

# a visitor the generator has not ended this long after its own limit (connection.job_timeout) is ended
# here: the generator answers by itself unless its thread is stuck, and then the flash would stay
VISITOR_MARGIN: float = 30.0
NO_CAMERA: str = 'no camera frames'

class CamType(Enum):
    VIMBA   = 2
    WEB     = 3

class Main():
    def __init__(self, width: int, height: int, portraitMode: bool = False, camType: CamType = CamType.VIMBA,
                 fullscreen: bool = False) -> None:
        # the first line of every log: what happened before this start
        logger.info(crash_summary())
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
        if fullscreen:
            # the installation's start: borderless (exclusive fullscreen minimizes when a player gets focus)
            self.settings.render.window.fullscreen_mode = FullscreenMode.WINDOWED_FULLSCREEN
        self.settings.initialize()
        self.settings_server = NiceServer(self.settings, self.settings.server, on_exit=self.stop)

        self.sequencer: Sequencer = Sequencer(self.settings.sequence)
        # the generator fills the player with the grin frames, the render plays them on its own clock
        self.player: GrinPlayer = GrinPlayer(self.settings.sequence)
        self.render: SmileRender = SmileRender(self.settings.render, self.sequencer, self.player, self.width, self.height)
        self.diffuse: Generator = Generator(self.settings.pipeline, self.player)
        # the face crop is the image ComfyUI generates from, so it has the workflow's portrait size
        portrait_width, portrait_height = self.diffuse.workflow.portrait_size
        self.face: FaceFinder = FaceFinder(self.settings.camera.face, portrait_width, portrait_height)

        # the camera device; the panel can switch to an image from input/ at any time
        if camType == CamType.VIMBA:
            from modules.cam.VimbaCam import VimbaCam
            device = VimbaCam(self.settings.camera)
            device_name = 'vimba'
        else:
            from modules.cam.WebCam import WebCam
            device = WebCam(self.settings.camera)
            device_name = 'webcam'
        self.camera: Camera = Camera(self.settings.camera, (self.width, self.height), device, device_name)
        self.camera.setRotate90(self.portraitMode)

        self.audio: AudioDB = AudioDB(self.settings.audio, 'data/audio/music_map.json')
        self.osc: Osc = Osc(self.settings.audio.osc)
        self.screen_test: ScreenTest = ScreenTest(self.settings.audio, bang=self._test_bang)  # the laughter test, stepped from sequence_callback
        self.save: ClipSaver = ClipSaver(self.settings.sequence)
        self._grin_saved: bool = False      # the visitor's grin was saved to its last frame
        self._saving: bool = False          # the visitor's files are open in the saver
        self._style_picked: bool = False    # the style of the next visitor is picked: until the flash
        self._face_frame_time: float = 0.0  # arrival of the camera frame the face finder got last
        self._visitor_start: float | None = None    # the capture of the visitor the generator is working on
        self.artnet = Artnet(self.settings.artnet, 'data/artnet.json')

        self._running: bool = False
        self._stopping: bool = False

        self.laststate: str = ''

    def start(self) -> None:
        self.camera.setColorCallback(self.render.set_cam_image)
        self.camera.open()
        self.camera.startCapture()

        self.diffuse.add_message_callback(self.render.set_diffusion_message)
        self.diffuse.add_message_callback(self.diffusion_callback)
        self.diffuse.start()

        self.face.add_frame_callback(self.face_callback)
        self.face.add_frame_callback(self.render.set_faces)
        self.face.start()

        self.osc.start()
        self.artnet.start()

        self.settings_server.start()

        # last: its frames drive the sequencer, which drives everything else
        self.render.add_exit_callback(self.stop)
        self.sequencer.add_message_callback(self.sequence_callback)
        self.render.start()

        self._running = True

    def stop(self) -> None:
        # called from the render window (Esc), the settings panel (Exit) and Ctrl+C; run once
        if self._stopping:
            return
        self._stopping = True

        # first: stops the sequencer callbacks
        steps = (self.sequencer.clear_message_callbacks, self.render.stop, self.sequencer.close,
                 self.artnet.stop, self.osc.stop,
                 self.camera.clearColorCallbacks, self.camera.close,
                 self.diffuse.clear_message_callbacks, self.diffuse.stop,
                 self.face.clear_frame_callbacks, self.face.stop,
                 self.save.stop,
                 self.settings_server.stop)
        try:
            for step in steps:
                # a step that fails must not keep the others, or the process, from ending
                try:
                    step()
                except Exception:
                    logger.exception('Error while stopping (%s)', getattr(step, '__qualname__', step))
        finally:
            self._running = False

    def _test_bang(self, index: int, sample: int) -> None:
        """The laughter test's bang to Max: the screen's index and the sample, past the patch's gates."""
        self.osc.send('/test_bang', (index, sample))

    def sequence_callback(self, message: SeqenceMessage) -> None:
        seq: SequenceType = message.sequence_type
        ani: AnimationType = message.animation_type
        state: str = AnimationTypeNames[message.animation_type]

        # no visitor while the generator can't take a capture (ComfyUI not there, or still warming up):
        # the flash would hold until it can. The operator sees why on the image.
        # None either while the raw camera is shown: the sequence would run unseen behind it.
        # Nor while the laughter test runs: a visitor's state would close Max's laughter gate
        ready: bool = self.diffuse.ready
        self.sequencer.blocked = not ready or self.settings.render.show_camera or self.settings.audio.laughter_test
        # the operator's line: the generator's reason first, then a camera that delivers no frames
        status: str | None = None if ready else self.settings.pipeline.connection.status
        if status is None and self.camera.stalled:
            status = NO_CAMERA
        self.render.set_status(status)
        self.render.set_warmup(self.diffuse.warmup)

        cam_image, cam_time = self.camera.frame()
        artnet_progress: float = message.progress
        if seq == SequenceType.TIMEOUT:
            artnet_progress = 1.0
        self.artnet.set_progress(artnet_progress, ani)

        if seq == SequenceType.CAMERA_DETECT and not self._style_picked:
            # one style per visitor; the style id is also the movement id that picks the music.
            # Picked when a face starts charging, so the music player has the track loaded at the
            # flash; a charge that drains and starts again keeps the style that was picked
            self._style_picked = True
            movement_id: str = self.diffuse.next_style()
            self.audio.update(movement_id)
        if seq == SequenceType.FLASH and message.initial:
            self._style_picked = False

        audio = self.settings.audio
        if audio.osc.active:
            progress: float = message.progress
            audio_id: str = self.audio.get_audio_id()
            self.osc.send('/music_id', audio_id)
            self.osc.send('/state', state)
            self.osc.send('/progress', progress)
            # the Max panel's controls: the panel is their source, sent every frame like the others
            # so a Max that restarted follows it within a frame. int(): python-osc sends a bool as
            # OSC True/False, which Max's toggles do not take
            self.osc.send('/sound', int(audio.sound))
            self.osc.send('/music_volume', audio.music_volume)
            self.osc.send('/laughter', int(audio.laughter))
            self.osc.send('/laughter_volume', audio.laughter_volume)
            self.osc.send('/laughter_offset', audio.laughter_offset)
            # the test closes the gate of the clips' laughs in the patch; its own bangs pass that gate
            self.osc.send('/laughter_test', int(audio.laughter_test))
        # the laughter test steps on the app clock too (the lights to the players, the test bangs to Max)
        self.screen_test.update(time.time())

        if seq == SequenceType.CAMERA_SHOW and message.initial:
            self.face.reset()

        if seq >= SequenceType.CAMERA_SHOW and seq <= SequenceType.CAMERA_DETECT and cam_time > self._face_frame_time:
            # new frames only, with the time they arrived: a camera that stopped then stops charging
            self._face_frame_time = cam_time
            self.face.set_image(cam_image, cam_time)

        if seq == SequenceType.CAPTURE:
            # the last frame of the recording in which the face qualified; the full camera frame without one
            cap_image: np.ndarray = cam_image
            portrait: np.ndarray | None = self.face.best()
            if portrait is not None:
                cap_image = portrait
            else:
                logger.warning('No face for the capture, using the camera frame')

            if ready:
                self.diffuse.set_image(cap_image)
                self._visitor_start = time.time()
            else:
                # ComfyUI went away during the flash: nobody would end it. The capture is the result
                logger.warning('Generator not ready at the capture, ending the visitor')
                self.sequencer.finish()
            self.render.set_cap_image(cap_image)
            self.save.new_sequence(cap_image, self.settings.pipeline.style_current)
            self._saving = True
            self._grin_saved = False
            # a test run on images: the next visitor is the next file
            self.camera.next_image()

        self._limit_visitor(seq)

        if seq == SequenceType.TIMEOUT:
            if message.initial == True:
                # a failed or stopped visitor has no complete grin: its folder is not kept
                self._saving = False
                self.save.end_sequence(keep=self._grin_saved)

        if seq == SequenceType.RESET:
            self.diffuse.interrupt()
            if self._saving:
                # a reset before the result: the visitor's files are not kept
                self._saving = False
                self.save.end_sequence(keep=False)

    def _limit_visitor(self, seq: SequenceType) -> None:
        """End a visitor the generator did not end: it does so itself at connection.job_timeout, so
        this only acts when its thread is stuck. Without it the flash, or the portrait, would stay."""
        if self._visitor_start is None:
            return
        now: float = time.time()
        grin = self.settings.pipeline.grin
        if not (SequenceType.CAPTURE <= seq <= SequenceType.DIFF_LOOP):
            self._visitor_start = None      # ended
        elif grin.loop or grin.still:
            self._visitor_start = now       # testing: the grin goes round until both are off
        elif now - self._visitor_start > self.settings.pipeline.connection.job_timeout + VISITOR_MARGIN:
            logger.error('The generator did not end the visitor after %.0f s, ending it', now - self._visitor_start)
            self._visitor_start = None
            self.diffuse.interrupt()
            self.sequencer.finish()

    def face_callback(self, frame: FaceFrame) -> None:
        # the sequencer charges toward the flash while the face qualifies
        self.sequencer.set_face(frame.qualified, frame.time)

    def diffusion_callback(self, message: GenMessage) -> None:
        # a visitor's recording: the portrait still and the grin clip (not the step previews)
        if message.image is None:
            return
        # the grin loop and still are for testing: their frames would fill the clip without end
        grin = self.settings.pipeline.grin
        if message.state == GenState.DIFF_LOOP and (grin.loop or grin.still):
            return
        if message.state == GenState.DIFFUSION and message.final:
            # the record first: it goes into the portrait and the clip
            self.save.set_record(self.diffuse.record())
            self.save.save_portrait(message.image)
        if message.state == GenState.DIFF_LOOP:
            self.save.save_frame(message.image)
            if message.progress >= 1.0:
                self._grin_saved = True

    def isRunning(self) -> bool :
        return self._running
