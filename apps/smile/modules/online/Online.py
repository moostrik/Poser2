"""The online turn: when the room's eye is free and someone waits at the relay, SMILE takes a turn.
The phone's frames go into `Camera` in place of the device, so the face finder, the charge, the flash
and the generation run as for a visitor inside; what differs is here: the judging goes back to the
phone (every face's box and distance, the charge), the transformation's images follow, and when the
visitor is saved its files are uploaded and the phone gets the result.

`update` runs once per render frame from Main.sequence_callback (the app clock, like the laughter
test) and is the only place the phase changes; the link's thread only leaves notes (`_notes`).

    OFF -> IDLE (the link is up) -> ASKING (turn_start sent) -> JUDGING (the phone has the camera)
    -> TRANSFORMING (the flash) -> DELIVERING (the files) -> IDLE

A turn is missed when no face qualified within `turn_window` (the charge is empty when it runs out;
a charge that is rising gets its chance), lost when the phone leaves, reset when the operator resets.
SMILE decides every turn's start and end; the relay keeps the line."""

import logging
import time
from enum import Enum, auto
from threading import Lock
from typing import Callable

import cv2
import numpy as np

from modules.cam.Camera import Camera
from modules.comfy.messages import GenMessage, GenState
from modules.comfy.settings import PipelineSettings, StyleSettings
from modules.face.FaceFinder import FaceFinder, FaceFrame
from modules.face.quality import rejection_label
from modules.online.Link import Link, read_token
from modules.online.settings import OnlineSettings
from modules.render.Sequencer import CAMERA_STATES, SeqenceMessage, Sequencer, SequenceType

logger = logging.getLogger(__name__)

ASK_TIMEOUT: float = 3.0        # s for the relay to answer a turn_start
SAVE_TIMEOUT: float = 20.0      # s for the saver to close the visitor's files after the result
EYE_INTERVAL: float = 1.0       # s between the eye messages to the relay
PREVIEW_INTERVAL: float = 0.25  # s between the sampler previews sent to the phone
PREVIEW_QUALITY: int = 70       # JPEG, the images for the phone's window
PREVIEW_WIDTH: int = 512

# the eye's status line as the phones see it (the installation's own language)
LOOKING: str = 'looking for faces'
INSIDE: str = 'looking at someone inside'
WARMING: str = 'warming up'
BUSY: str = 'busy inside'

LinkFactory = Callable[[str, str, Callable[[dict], None], Callable[[bytes], None]], Link]
Caption = Callable[[], tuple[str, str]]     # the credit: name and the visitor's line
Record = Callable[[], dict]                 # Generator.record


class Phase(Enum):
    OFF = auto()
    IDLE = auto()
    ASKING = auto()
    JUDGING = auto()
    TRANSFORMING = auto()
    DELIVERING = auto()


def core_description(prompt: str, prefix: str, suffix: str) -> str:
    """QwenVL's own words: the prompt without the style's prefix and suffix around them."""
    text: str = prompt.strip()
    prefix = prefix.strip()
    suffix = suffix.strip()
    if prefix and text.startswith(prefix):
        text = text[len(prefix):]
    if suffix and text.endswith(suffix):
        text = text[:-len(suffix)]
    return text.strip()


def encode(image: np.ndarray, width: int = PREVIEW_WIDTH, quality: int = PREVIEW_QUALITY) -> bytes | None:
    """A JPEG for the phone's window, no wider than `width`."""
    h, w = image.shape[:2]
    if w > width:
        image = cv2.resize(image, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpeg.tobytes() if ok else None


def face_message(frame: FaceFrame, charge: float) -> dict:
    return {'type': 'face', 'charge': round(charge, 3), 'faces': [
        {'box': [round(v, 4) for v in f.crop], 'distance': f.distance, 'subject': f.subject,
         'reject': rejection_label(f) if f.rejection is not None else None} for f in frame.faces]}


class Online:
    def __init__(self, settings: OnlineSettings, pipeline: PipelineSettings, camera: Camera, sequencer: Sequencer,
                 face: FaceFinder, record: Record, caption: Caption, link_factory: LinkFactory = Link) -> None:
        self.settings: OnlineSettings = settings
        self.pipeline: PipelineSettings = pipeline
        self.camera: Camera = camera
        self.sequencer: Sequencer = sequencer
        self.face: FaceFinder = face
        self._record: Record = record
        self._caption: Caption = caption
        self._link_factory: LinkFactory = link_factory
        self._link: Link | None = None
        self._phase: Phase = Phase.OFF
        self._notes_mutex: Lock = Lock()
        self._notes: dict = {}          # from the link's and the saver's threads, taken in update()
        self._waiting: int = 0
        self._free_since: float | None = None
        self._asked: float = 0.0
        self._turn_start: float = 0.0
        self._charge: float = 0.0
        self._seq: SequenceType = SequenceType.NONE
        self._eye_sent: float = 0.0
        self._eye_last: tuple[bool, str] | None = None
        self._preview_sent: float = 0.0
        self._described: bool = False
        self._grin_told: bool = False
        self._deliver_start: float = 0.0
        self._delivering: bool = False  # an upload runs
        self._deliver_id: str = ''
        self._no_token_warned: bool = False
        # the switch is saved with the presets: the readouts start clean
        settings.connected = False
        settings.status = 'off'
        settings.waiting = 0
        settings.turns = 0

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def turn(self) -> bool:
        """A turn runs: the installation is someone outside's."""
        return self._phase in (Phase.JUDGING, Phase.TRANSFORMING, Phase.DELIVERING)

    def stop(self) -> None:
        self._end_turn(None)
        self._stop_link()

    # FROM THE OTHER THREADS: notes only, taken on the app clock
    def _note(self, **values) -> None:
        with self._notes_mutex:
            self._notes.update(values)

    def _take_notes(self) -> dict:
        with self._notes_mutex:
            notes, self._notes = self._notes, {}
            return notes

    def _on_message(self, message: dict) -> None:
        kind: str = str(message.get('type', ''))
        if kind == 'line':
            self._note(waiting=int(message.get('waiting', 0)))
        elif kind == 'turn':
            self._note(turn=message.get('id') or '')
        elif kind == 'gone':
            self._note(gone=True)

    def _on_frame(self, jpeg: bytes) -> None:
        if self._phase != Phase.JUDGING:
            return
        image: np.ndarray | None = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            self.camera.push_online(image)

    def on_face(self, frame: FaceFrame) -> None:
        """Every detector frame (Main.face_callback): the judging goes to the phone during its turn."""
        if self._phase == Phase.JUDGING and self._link is not None:
            self._link.send(face_message(frame, self._charge))

    def on_generation(self, message: GenMessage) -> None:
        """The generator's messages (Main.diffusion_callback): the transformation goes to the phone."""
        if self._phase != Phase.TRANSFORMING or self._link is None:
            return
        link: Link = self._link
        if message.state == GenState.CAM_DEPTH:
            link.send({'type': 'stage', 'stage': 'depth'})
            self._show(message.image)
        elif message.state == GenState.DIFFUSION:
            if not self._described:
                self._described = True
                link.send({'type': 'stage', 'stage': 'describe', 'text': self.description()})
            if message.final:
                link.send({'type': 'stage', 'stage': 'portrait'})
                self._show(message.image, force=True)
            else:
                if message.progress > 0.0:
                    link.send({'type': 'stage', 'stage': 'diffusion', 'progress': round(message.progress, 3)})
                self._show(message.image)
        elif message.state == GenState.DIFF_LOOP and not self._grin_told:
            self._grin_told = True
            link.send({'type': 'stage', 'stage': 'grin'})

    def on_saved(self, id: str, style: str, kept: bool, grin: str, portrait: str) -> None:
        """The saver closed the visitor's files (ClipSaver's callback, on its thread)."""
        self._note(saved=(id, style, kept, grin, portrait))

    def _show(self, image: np.ndarray | None, force: bool = False) -> None:
        if image is None or self._link is None:
            return
        now: float = time.time()
        if not force and now - self._preview_sent < PREVIEW_INTERVAL:
            return
        self._preview_sent = now
        jpeg: bytes | None = encode(image)
        if jpeg is not None:
            self._link.send_image(jpeg)

    def description(self) -> str:
        style: StyleSettings = self.pipeline.style_b if self.pipeline.style_current == 'B' else self.pipeline.style_a
        return core_description(str(self._record().get('description', '')), style.prefix, style.suffix)

    # THE APP CLOCK
    def update(self, now: float, message: SeqenceMessage, ready: bool, room_busy: bool) -> None:
        """Once per render frame. `ready` is the generator, `room_busy` the operator's states that
        block the room too (the raw camera shown, the laughter test)."""
        s: OnlineSettings = self.settings
        seq: SequenceType = message.sequence_type
        if seq == SequenceType.CAMERA_DETECT:
            self._charge = message.progress
        elif seq in CAMERA_STATES:
            self._charge = 0.0
        notes: dict = self._take_notes()
        if 'waiting' in notes:
            self._waiting = notes['waiting']
            s.waiting = self._waiting

        if not s.active:
            if self._phase != Phase.OFF:
                self._end_turn('reset')
                self._stop_link()
                self._phase = Phase.OFF
                s.status = 'off'
            self._seq = seq
            return
        if self._link is None and not self._start_link():
            self._seq = seq
            return
        link: Link = self._link       # type: ignore[assignment]
        connected: bool = link.connected
        if s.connected != connected:
            s.connected = connected
        if self._phase == Phase.OFF:
            self._phase = Phase.IDLE

        # the eye, as the phones see it
        free: bool = self._eye_free(now, seq, ready, room_busy, connected)
        self._send_eye(now, link, free, self._eye_status(seq, ready, room_busy))

        if not connected:
            if self.turn:
                logger.warning('online: the relay is gone during a turn')
                self._end_turn(None)
            if self._phase != Phase.IDLE:
                self._phase = Phase.IDLE
            s.status = 'connecting'
        elif self._phase == Phase.IDLE:
            s.status = f'idle ({self._waiting} waiting)'
            if free and self._waiting > 0:
                link.send({'type': 'turn_start', 'window': s.turn_window})
                self._asked = now
                self._phase = Phase.ASKING
        elif self._phase == Phase.ASKING:
            if 'turn' in notes:
                if notes['turn']:
                    self._begin_turn(now)
                else:
                    self._phase = Phase.IDLE
            elif now - self._asked > ASK_TIMEOUT:
                logger.warning('online: the relay did not answer the turn')
                self._phase = Phase.IDLE
        elif self._phase == Phase.JUDGING:
            s.status = 'turn: judging'
            if notes.get('gone'):
                self._end_turn('lost')
            elif seq == SequenceType.RESET:
                self._end_turn('reset')
            elif seq not in CAMERA_STATES:
                # the flash: the capture is taken from the phone's frames, the transformation is the machine's
                link.send({'type': 'stage', 'stage': 'flash'})
                self._phase = Phase.TRANSFORMING
                self._described = False
                self._grin_told = False
            elif seq == SequenceType.CAMERA_SHOW and now - self._turn_start > s.turn_window:
                self._end_turn('missed')
        elif self._phase == Phase.TRANSFORMING:
            s.status = 'turn: transforming'
            if self.camera.online and seq >= SequenceType.ASK:
                # the capture is taken: the room's camera is back underneath, unseen until the result fades
                self.camera.set_online(False)
            if seq == SequenceType.RESET:
                self._end_turn('reset')
            elif (seq == SequenceType.TIMEOUT and message.initial) or seq in CAMERA_STATES:
                # the result stands on the screens: the saver closes the files now
                self._phase = Phase.DELIVERING
                self._deliver_start = now
                self._delivering = False
        elif self._phase == Phase.DELIVERING:
            s.status = 'turn: delivering'
            if self._delivering:
                if notes.get('delivered') is not None:
                    self._finish_turn(bool(notes['delivered']))
            elif 'saved' in notes:
                id, style, kept, grin, portrait = notes['saved']
                if kept:
                    self._delivering = True
                    self._deliver_id = id
                    files: list[tuple[str, str, str]] = [('portrait', id, portrait), ('still', id, portrait), ('grin', id, grin)]
                    link.upload(files, lambda ok: self._note(delivered=ok))
                else:
                    logger.warning('online: the visitor %s was not saved, no result', id)
                    self._end_turn('reset')
            elif now - self._deliver_start > SAVE_TIMEOUT:
                logger.warning('online: no saved files within %.0f s, no result', SAVE_TIMEOUT)
                self._end_turn('reset')
        self._seq = seq

    # THE EYE
    def _eye_free(self, now: float, seq: SequenceType, ready: bool, room_busy: bool, connected: bool) -> bool:
        """Free for a few seconds: nobody inside, the generator ready, no turn, not paused."""
        free_now: bool = (connected and ready and not room_busy and seq == SequenceType.CAMERA_SHOW
                          and not self.sequencer.settings.pause and self._phase == Phase.IDLE)
        if not free_now:
            self._free_since = None
            return False
        if self._free_since is None:
            self._free_since = now
        return now - self._free_since >= self.settings.eye_free_after

    @staticmethod
    def _eye_status(seq: SequenceType, ready: bool, room_busy: bool) -> str:
        if not ready:
            return WARMING
        if room_busy:
            return BUSY
        if seq == SequenceType.CAMERA_SHOW:
            return LOOKING
        return INSIDE

    def _send_eye(self, now: float, link: Link, free: bool, status: str) -> None:
        if not link.connected:
            self._eye_last = None
            return
        eye: tuple[bool, str] = (free, status)
        if eye != self._eye_last or now - self._eye_sent > EYE_INTERVAL:
            self._eye_last = eye
            self._eye_sent = now
            link.send({'type': 'eye', 'free': free, 'status': status})

    # THE TURN
    def _begin_turn(self, now: float) -> None:
        self._turn_start = now
        self._phase = Phase.JUDGING
        self._charge = 0.0
        self.face.reset()               # the room's last faces are not this visitor's
        self.camera.set_online(True)
        self.settings.turns += 1
        logger.info('online: turn %d starts', self.settings.turns)

    def _end_turn(self, reason: str | None) -> None:
        """Back to the room: the camera, and the phone is told why (None: nothing can be sent)."""
        if not self.turn and self._phase != Phase.ASKING:
            return
        self.camera.set_online(False)
        if reason is not None and self._link is not None:
            self._link.send({'type': 'turn_end', 'reason': reason})
        if self.turn:
            logger.info('online: turn ends (%s)', reason)
        self._phase = Phase.IDLE
        self._delivering = False

    def _finish_turn(self, uploaded: bool) -> None:
        if uploaded and self._link is not None:
            name, line = self._caption()
            self._link.send({'type': 'result', 'id': self._deliver_id, 'name': name, 'caption': line})
            self._end_turn('done')
        else:
            self._end_turn('reset')

    # THE LINK
    def _start_link(self) -> bool:
        token: str = read_token(self.settings.token_file)
        if not token:
            if not self._no_token_warned:
                self._no_token_warned = True
                logger.warning('online: no token in %s: write one line into it (the relay has the same)', self.settings.token_file)
            self.settings.status = f'no token in {self.settings.token_file}'
            return False
        self._no_token_warned = False
        self._link = self._link_factory(self.settings.relay, token, self._on_message, self._on_frame)
        self._link.start()
        return True

    def _stop_link(self) -> None:
        link, self._link = self._link, None
        if link is not None:
            link.stop()
        self.settings.connected = False
        self._eye_last = None
