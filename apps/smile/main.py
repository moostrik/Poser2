"""SMILE: a camera captures a face, an image generator turns it into a portrait, shown with music and lights,
and the grins of the visitors before play on the two walls of the sphere (README.md, SMILE.md)."""

from .cam import Camera
from .view import cover
from modules import recording
from modules.recording import replay
from modules.camera import CameraFrame, CameraSource, upright_image, vimba, webcam
from modules.session import Session
from modules.inference import GpuImage, SerialUploader
from modules.inference.face import FaceDetector
from .board import OnlineCell, RenderBoard
from .face import FaceFinder, FaceObservations
from .render import SmileRender
from .sequence import CAMERA_STATES, AnimationType, AnimationTypeNames, GrinPlayer, Sequencer, SequenceMessage, SequenceType
from .comfy import Generator, GeneratorMessage, GeneratorState
from modules.inout import OscSender
from .audio import LaughterTest, MusicMap, laugh_files
from .credit import CreditLines
from .credit_scroll import CreditScroller
from .save import ClipSaver
from .lights import Lights
from .online import Online, http_base
from .player import Player, online_cell_mode
from . import sphere
from modules.settings import SettingsStore
from modules.panel import PanelServer
from .settings import Settings
from argparse import ArgumentParser, Namespace
from enum import Enum
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)

DATA_PATH = 'apps/smile/data'

# a visitor the generator has not ended this long after its own limit (connection.job_timeout) is ended
# here: the generator answers by itself unless its thread is stuck, and then the flash would stay
VISITOR_MARGIN: float = 30.0
NO_CAMERA: str = 'no camera frames'

class CamType(Enum):
    VIMBA   = 2
    WEB     = 3


def parse_args(args: list[str]) -> Namespace:
    """The app's own arguments, after the launcher's: `launcher.py -app smile -fs -wc`."""
    parser: ArgumentParser = ArgumentParser(prog='launcher.py -app smile', description='SMILE')
    parser.add_argument('-dw',  '--width',          type=int,   default=1280,   help='image width (4:5 like the generated portraits; 1280 x 1024 is the display in portrait mode)')
    parser.add_argument('-dh',  '--height',         type=int,   default=1024,   help='image height')
    parser.add_argument('-wc',  '--webcam',         action='store_true',        help='use a webcam instead of the Vimba camera')
    parser.add_argument('-lm',  '--landscape',      action='store_true',        help='landscape instead of portrait mode')
    parser.add_argument('-fs',  '--fullscreen',     action='store_true',        help='open borderless fullscreen, the walls too (the installation) instead of the window modes of the settings file')
    return parser.parse_args(args)


class SmileMain():
    def __init__(self, simulation: bool = False, interface: bool = True, args: list[str] = []) -> None:
        self.is_running: bool = False
        self.is_finished: bool = False
        options: Namespace = parse_args(args)
        self.width: int = round(options.width / 4) * 4
        self.height: int = round(options.height / 4) * 4
        self.portrait_mode: bool = not options.landscape
        if self.portrait_mode:
            self.width, self.height = self.height, self.width
        cam_type: CamType = CamType.WEB if options.webcam else CamType.VIMBA
        fullscreen: bool = options.fullscreen

        # SETTINGS
        self.settings = Settings()
        self.settings.setup.panel.title = 'SMILE'
        self.settings.setup.panel.port = 667
        self.store = SettingsStore(f'{DATA_PATH}/settings', default_startup='Smile')
        self.store.load_startup(self.settings)
        if not interface:
            self.settings.setup.panel.browser = False  # launcher override; the settings file cannot turn it back on
        if fullscreen:
            # the installation's start: borderless (exclusive fullscreen minimizes when a wall gets focus), the walls too
            for window in self.settings.setup.windows.windows:
                window.fullscreen = True
        self.settings.initialize()
        self.panel_server = PanelServer(self.settings, self.settings.setup.panel, self.store, on_exit=self.stop)

        self.sequencer: Sequencer = Sequencer(self.settings.setup.sequence)
        # the generator fills the grin player with the grin frames, the render plays them on its own clock
        self.grin_player: GrinPlayer = GrinPlayer(self.settings.setup.sequence.grin)
        # what the app's threads make for the render, read at its tick
        self.board: RenderBoard = RenderBoard()
        self.render: SmileRender = SmileRender(self.board, self.settings.setup.render, self.settings.setup.windows, self.settings.setup.walls,
                                               self.sequencer, self.grin_player, self.width, self.height)
        # the clip player of the sphere's walls: one pool for the 48 cells of the render's two walls
        self.clip_player: Player = Player(self.settings.setup.walls, list(self.render.walls),
                                          CreditLines(str(self.settings.setup.render.credit_lines)))
        self.generator: Generator = Generator(self.settings.setup.generator, self.grin_player)
        # the face crop is the image ComfyUI generates from, so it has the workflow's portrait size
        portrait_width, portrait_height = self.generator.workflow.portrait_size
        # the faces as the other apps find them: every camera frame uploaded (the render shows it from the board), the
        # faces found on the GPU while the camera states run, and smile's choice of a subject among them
        self.uploader: SerialUploader = SerialUploader(self.settings.setup.camera.uploader)
        self.face_detector: FaceDetector = FaceDetector(self.settings.setup.camera.face_detector)
        self.face: FaceFinder = FaceFinder(self.settings.setup.camera.face, self.settings.setup.camera.face_detector, portrait_width, portrait_height,
                                           (self.width, self.height))

        # the camera device; the panel can switch to an image from input/ at any time. -sim plays a recording in its place
        self.recording_player: recording.Player | None = None
        device: CameraSource
        if simulation:
            self.recording_player = recording.Player(self.settings.setup.camera.simulator, data_path=DATA_PATH)
            device = replay.Camera(self.recording_player, self.settings.setup.camera.replay, None, None)
            device_name = 'replay'
        elif cam_type == CamType.VIMBA:
            # without the Vimba 6 SDK (VIMBA_HOME, VimbaC) the backend logs an error at start: no device, the image still works
            device = vimba.Camera(self.settings.setup.camera, rotate90=self.portrait_mode)
            device_name = 'vimba'
        else:
            device = webcam.Camera(self.settings.setup.camera, rotate90=self.portrait_mode)
            device_name = 'webcam'
        self.camera: Camera = Camera(self.settings.setup.camera, device, device_name)
        # the device's frames as they came, recorded after the live path took them, in chunks the session splits
        self.recorder: recording.Recorder = recording.Recorder(self.settings.setup.camera.recorder, data_path=DATA_PATH)
        self.session: Session = Session(self.settings.setup.camera.control)
        self.camera.add_device_frame_callback(self.recorder.submit_frame)

        self.music_map: MusicMap = MusicMap(self.settings.setup.audio, f'{DATA_PATH}/audio/music_map.json')
        self.osc: OscSender = OscSender(self.settings.setup.audio.osc, events=('/test_bang',))
        # the laughter test, stepped from sequence_callback: it lights the walls' cells and bangs Max
        self.laughter_test: LaughterTest = LaughterTest(self.settings.setup.audio, send=self.clip_player.light, bang=self._test_bang)
        # the name going round the walls while nobody is seen, planned here and handed to them every second
        self.credit_scroller: CreditScroller = CreditScroller(self.settings.setup.render, self.clip_player.scroll_loop)
        self._face_seen: float = time.perf_counter()    # the last qualifying face or visitor: the name waits from here
        self.save: ClipSaver = ClipSaver(self.settings.setup.sequence.saver, self.settings.setup.sequence.grin,
                                         watermark=self.settings.setup.render.credit_name)
        self._grin_saved: bool = False      # the visitor's grin was saved to its last frame
        self._saving: bool = False          # the visitor's files are open in the saver
        self._style_picked: bool = False    # the style of the next visitor is picked: until the flash
        self._judging: bool = False         # the camera states: the camera's frames go to the faces (the render's thread sets it)
        self._visitor_start: float | None = None    # the capture of the visitor the generator is working on
        self._flash_time: float = 0.0       # when the flash started: the music starts then (Max, the phone, the clip's sound)
        self.lights: Lights = Lights(self.settings.setup.artnet, f'{DATA_PATH}/artnet.json')
        # SMILE Online: a turn for someone outside when the room's eye is free, stepped from sequence_callback
        self.online: Online = Online(self.settings.setup.online, self.camera, self.sequencer, self.face, self.render.grab,
                                     lambda: self.settings.setup.render.credit_name,
                                     lambda: (self.music_map.get_music_id(), self.music_map.track_path()),
                                     laughs=laugh_files, laughter=lambda: self.settings.setup.audio.laughter, pick=self.pick_style)
        self.save.add_saved_callback(self.online.on_saved)

        self._stopping: bool = False

    def start(self) -> None:
        self.recorder.start()
        self.camera.add_frame_callback(self.uploader.submit_frame)
        self.camera.start()
        if self.recording_player is not None:
            self.recording_player.start()       # after the replay camera hooked into it

        self.generator.add_message_callback(self._show_generator_message)
        self.generator.add_message_callback(self.generator_callback)
        self.generator.start()

        self.face.add_frame_callback(self.face_callback)
        self.face.add_frame_callback(self.board.set_faces)
        self.face_detector.add_face_callback(self.face.submit)
        self.uploader.add_gpu_image_callback(self._judge_image)
        self.uploader.add_finished_gpu_image_callback(self.board.set_camera_image)
        self.face.start()
        self.face_detector.start()
        self.uploader.start()

        self.osc.start()
        self.lights.start()

        self.panel_server.start()

        # last: the render's frames drive the sequencer, which drives everything else; the launcher runs it after start()
        self.render.add_exit_callback(self.stop)
        self.sequencer.add_message_callback(self.sequence_callback)
        if any(window.enabled for window in self.settings.setup.windows.windows[1:]):
            self.clip_player.start()
        else:
            logger.info('No wall window shown in the settings file (windows.window_1, window_2): the clip player stays off')

        self.is_running = True

    def stop(self) -> None:
        # called from the render window (Esc), the settings panel (Exit) and Ctrl+C; run once
        if self._stopping:
            return
        self._stopping = True

        # first: stops the sequencer callbacks
        steps = (self.sequencer.clear_message_callbacks, self.render.stop, self.clip_player.stop, self.sequencer.close,
                 self.online.stop, self.lights.stop, self.osc.stop,
                 *((self.recording_player.stop,) if self.recording_player is not None else ()),
                 self.camera.stop, self.recorder.stop,
                 self.generator.clear_message_callbacks, self.generator.stop,
                 self.uploader.stop, self.face_detector.stop,
                 self.face.clear_frame_callbacks, self.face.stop,
                 self.save.stop,
                 self.panel_server.stop)
        try:
            for step in steps:
                # a step that fails must not keep the others, or the process, from ending
                try:
                    step()
                except Exception:
                    logger.exception('Error while stopping (%s)', getattr(step, '__qualname__', step))
        finally:
            self.is_running = False
            self.is_finished = True

    def _test_bang(self, index: int, sample: int) -> None:
        """The laughter test's bang to Max: the screen's index and the sample, past the patch's gates."""
        self.osc.submit('/test_bang', (index, sample))

    def sequence_callback(self, message: SequenceMessage) -> None:
        seq: SequenceType = message.sequence_type
        ani: AnimationType = message.animation_type
        state: str = AnimationTypeNames[message.animation_type]

        # no visitor while the generator can't take a capture (ComfyUI not there, or still warming up):
        # the flash would hold until it can. The operator sees why on the image.
        # None either while the raw camera is shown: the sequence would run unseen behind it.
        # Nor while the laughter test runs: a visitor's state would close Max's laughter gate
        ready: bool = self.generator.ready
        self.sequencer.blocked = not ready or self.settings.setup.render.show_camera or self.settings.setup.audio.laughter_test
        # the operator's line: the generator's reason first, then a camera that delivers no frames
        status: str | None = None if ready else self.settings.setup.generator.connection.status
        if status is None and self.camera.stalled:
            status = NO_CAMERA
        self.board.set_status(status)
        self.board.set_warmup(self.generator.warmup)

        artnet_progress: float = message.progress
        if seq == SequenceType.TIMEOUT:
            artnet_progress = 1.0
        self.lights.set_progress(artnet_progress, ani)

        if seq == SequenceType.CAMERA_DETECT:
            self.pick_style()
        if seq == SequenceType.FLASH and message.initial:
            self._style_picked = False
            self._flash_time = time.perf_counter()
        if seq == SequenceType.DIFF_LOOP and message.initial:
            # the grin starts playing: the share clip gets the track from the point the room hears now
            self.save.submit_share(self.music_map.track_path(), time.perf_counter() - self._flash_time)

        audio = self.settings.setup.audio
        if audio.osc.enabled:
            progress: float = message.progress
            music_id: str = self.music_map.get_music_id()
            self.osc.submit('/music_id', music_id)
            self.osc.submit('/state', state)
            self.osc.submit('/progress', progress)
            # the Max panel's controls: the panel is their source, sent every frame like the others
            # so a Max that restarted follows it within a frame. int(): python-osc sends a bool as
            # OSC True/False, which Max's toggles do not take
            self.osc.submit('/sound', int(audio.sound))
            self.osc.submit('/music_volume', audio.music_volume)
            self.osc.submit('/laughter', int(audio.laughter))
            self.osc.submit('/laughter_volume', audio.laughter_volume)
            self.osc.submit('/laughter_offset', audio.laughter_offset)
            # the test closes the gate of the clips' laughs in the patch; its own bangs pass that gate
            self.osc.submit('/laughter_test', int(audio.laughter_test))
        # the laughter test steps on the app clock too (the lights on the walls, the test bangs to Max), as do the
        # scrolling name and the online turn: `perf_counter`, the walls' clock as well
        now: float = time.perf_counter()
        self.laughter_test.update(now)
        # and the scrolling name's passes, timed on the app clock too; only after `credit_scroll_idle` s without a
        # qualifying face (the finder sees no frames during a visitor, who counts as one) and not while the test lights the screens
        if seq not in CAMERA_STATES:
            self._face_seen = time.perf_counter()
        waiting: bool = time.perf_counter() - self._face_seen < self.settings.setup.render.credit_scroll_idle
        self.credit_scroller.update(now, self.settings.setup.audio.laughter_test or waiting)
        # the online turn: it takes the camera when the room's eye is free, and the sequence runs as for a visitor inside
        self.online.update(now, message, ready, self.settings.setup.render.show_camera or self.settings.setup.audio.laughter_test)
        self._online_cell()

        if seq == SequenceType.CAMERA_SHOW and message.initial:
            self.face.reset()

        # the camera states judge the camera's frames as they come (`_judge_image`)
        self._judging = SequenceType.CAMERA_SHOW <= seq <= SequenceType.CAMERA_DETECT

        if seq == SequenceType.CAPTURE:
            # the last frame of the recording in which the face qualified; without one the view of the newest camera
            # frame, turned, flipped and fitted here, once
            cap_image: np.ndarray = cover(upright_image(self.camera.frame()), self.width, self.height)
            portrait: np.ndarray | None = self.face.best()
            if portrait is not None:
                cap_image = portrait
            else:
                logger.warning('No face for the capture, using the camera frame')

            if ready:
                self.generator.set_image(cap_image)
                self._visitor_start = time.perf_counter()
            else:
                # ComfyUI went away during the flash: nobody would end it. The capture is the result
                logger.warning('Generator not ready at the capture, ending the visitor')
                self.sequencer.finish()
            # the capture is the first diffusion image too, until the generator sends one: without that, a failed
            # generation would blend the previous visitor's last frame in
            self.board.set_capture(cap_image)
            self.board.set_diffusion_image(cap_image)
            self.save.submit_open(cap_image, self.settings.setup.generator.style_current)
            self._saving = True
            self._grin_saved = False
            # a test run on images: the next visitor is the next file
            self.camera.next_image()

        self._limit_visitor(seq)

        if seq == SequenceType.TIMEOUT:
            if message.initial == True and self._saving:
                # still open here: a failed or stopped visitor has no complete grin, its folder is not kept
                self._saving = False
                self.save.submit_close(keep=self._grin_saved)

        if seq == SequenceType.RESET:
            self.generator.interrupt()
            if self._saving:
                # a reset before the result: the visitor's files are not kept
                self._saving = False
                self.save.submit_close(keep=False)

    def _online_cell(self) -> None:
        """The online cell of the walls, every frame: the screen `online.screen` shows the QR code to the phone
        page while SMILE Online waits for someone, the main screen during a turn, a clip like any cell otherwise."""
        online = self.settings.setup.online
        screen: int = int(online.screen)
        mode: str = 'clip'
        wall = cell = 0
        pass_value: str = self.online.pass_value
        if 1 <= screen <= sphere.SCREENS:
            wall, cell = sphere.cell_of_screen(screen)
            # the card waits for the relay's pass: a code without one would open the expired page
            mode = online_cell_mode(bool(online.active), bool(online.connected) and bool(pass_value), self.online.showing)
        self.clip_player.online_cell = wall * sphere.CELLS + cell if mode != 'clip' else None
        page: str = str(online.page).strip() or http_base(str(online.relay)) + '/'
        if pass_value:
            page += ('&' if '?' in page else '?') + 'pass=' + pass_value
        self.board.set_online_cell(OnlineCell(wall, cell, mode, page) if mode != 'clip' else None)

    def _limit_visitor(self, seq: SequenceType) -> None:
        """End a visitor the generator did not end: it does so itself at connection.job_timeout, so
        this only acts when its thread is stuck. Without it the flash, or the portrait, would stay."""
        if self._visitor_start is None:
            return
        now: float = time.perf_counter()
        grin = self.settings.setup.generator.grin
        if not (SequenceType.CAPTURE <= seq <= SequenceType.DIFF_LOOP):
            self._visitor_start = None      # ended
        elif grin.loop or grin.still:
            self._visitor_start = now       # testing: the grin goes round until both are off
        elif now - self._visitor_start > self.settings.setup.generator.connection.job_timeout + VISITOR_MARGIN:
            logger.error('The generator did not end the visitor after %.0f s, ending it', now - self._visitor_start)
            self._visitor_start = None
            self.generator.interrupt()
            self.sequencer.finish()

    def pick_style(self) -> None:
        """One style per visitor; the style id also picks the music. Picked when a
        face starts charging (an online turn picks at its start), so the music player has the track loaded
        at the flash; a charge that drains and starts again keeps it."""
        if not self._style_picked:
            self._style_picked = True
            style_id: str = self.generator.next_style()
            self.music_map.set_style(style_id)

    def _judge_image(self, image: GpuImage) -> None:
        """An uploaded camera frame, on the uploader's thread: to the faces while the camera states run. Only new
        frames come, so a camera that stopped stops charging."""
        if self._judging:
            self.face_detector.submit(image)

    def _show_generator_message(self, message: GeneratorMessage) -> None:
        """A generator message, on its thread: its image to the board first, so the render takes it no later than the
        frame the new state starts, then the sequence's step (the capture's depth starts the reveal)."""
        if message.state == GeneratorState.CAM_DEPTH:
            if message.image is not None:
                self.board.set_capture_depth(message.image)
            self.sequencer.start_reveal()
        elif message.state == GeneratorState.DIFFUSION:
            if message.image is not None:
                self.board.set_diffusion_image(message.image)
            self.sequencer.set_sequence(SequenceType.DIFFUSION)
            self.sequencer.set_diffusion_progress(message.progress)
        elif message.state == GeneratorState.IDLE:
            if message.image is not None:
                self.board.set_diffusion_depth(message.image)
            self.sequencer.finish()

    def face_callback(self, frame: FaceObservations) -> None:
        # the sequencer charges toward the flash while the face qualifies
        self.sequencer.set_face(frame.qualified, frame.timestamp)
        if frame.qualified:
            self._face_seen = frame.timestamp
        self.online.on_face(frame)

    def generator_callback(self, message: GeneratorMessage) -> None:
        # a visitor's recording: the portrait still and the grin clip (not the step previews)
        if message.image is None:
            return
        # the grin loop and still are for testing: their frames would fill the clip without end
        grin = self.settings.setup.generator.grin
        if message.state == GeneratorState.DIFF_LOOP and (grin.loop or grin.still):
            return
        if message.state == GeneratorState.DIFFUSION and message.final:
            # the record first: it goes into the portrait and the clip
            self.save.submit_record(self.generator.record())
            self.save.submit_portrait(message.image)
        if message.state == GeneratorState.DIFF_LOOP:
            self.save.submit_frame(message.image)
            if message.progress >= 1.0 and not self._grin_saved:
                # the grin is complete: the recording closes now, while the grin still plays on the screens,
                # so the share clip is built and on its way to a visitor outside before the turn is over
                self._grin_saved = True
                if self._saving:
                    self._saving = False
                    self.save.submit_close(keep=True)

