"""The online turn: when the room's eye is free and someone waits at the relay, SMILE takes a turn.
The phone's frames go into `Camera` in place of the device, so the face finder, the charge, the flash
and the generation run as for a visitor inside; what differs is here: the judging goes back to the
phone (every face's box and distance, the charge), and from the flash to the end of the timeout the
phone gets the main screen as it is rendered (`ScreenGrab`, a few frames a second through the link),
so it shows what the room shows, the visitor's track to play from the flash (`music` at the turn's start: the
relay keeps the tracks, SMILE uploads one the first time it is used) and the sequence's stages with their progress, so the
phone plays the track as the Max patch does (the spin-up at the flash, the wind-down in the fade) and
laughs as the room does (the samples, uploaded once too). The turn ends when the sequence is back at the
camera, after the bloom and the fade; the visitor's saved files are uploaded after it (`Delivery`), and the
phone gets the result, addressed by its id, when they are there.

`update` runs once per render frame from Main.sequence_callback (the app clock, like the laughter
test) and is the only place the phase changes; the link's and the saver's threads only leave notes (`_notes`).

    OFF -> IDLE (the link is up) -> ASKING (turn_start sent) -> JUDGING (the phone has the camera)
    -> TRANSFORMING (the flash, the screen streams) -> IDLE (the delivery goes on beside the next turn)

A turn is missed when no face qualified within `turn_window` (the charge is empty when it runs out;
a charge that is rising gets its chance), lost when the phone leaves, reset when the operator resets.
SMILE decides every turn's start and end; the relay keeps the line."""

import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
from threading import Lock
from typing import Callable

import cv2
import numpy as np

from ..cam import Camera
from modules.face import FaceFinder, FaceFrame, rejection_label
from .link import Link, http_base, read_token
from .settings import OnlineSettings
from ..render import CAMERA_STATES, ScreenGrab, SequenceMessage, Sequencer, SequenceType

logger = logging.getLogger(__name__)

ASK_TIMEOUT: float = 3.0        # s for the relay to answer a turn_start
SAVE_TIMEOUT: float = 20.0      # s for the saver to close the visitor's files after the turn
EYE_INTERVAL: float = 1.0       # s between the eye messages to the relay
STAGE_INTERVAL: float = 0.25    # s between the stage progress messages to the phone
STILL_FPS: float = 2.0          # the stream while the result stands still (the timeout): the link is the clip's then
# share of the panel's values) so the link stays free for the messages and the clip; a short one lets them grow again

# the sequence's stages as the phone's tape follows them (the Max patch's states, see p speedVar there)
STAGE_NAMES: dict[SequenceType, str] = {
    SequenceType.FLASH: 'flash_in', SequenceType.CAPTURE: 'flash_in', SequenceType.ASK: 'flash_in',
    SequenceType.CONTROL_0: 'flash_out', SequenceType.CONTROL_1: 'flash_out', SequenceType.CONTROL_2: 'flash_out',
    SequenceType.DIFFUSION: 'diffusion', SequenceType.DIFF_LOOP: 'grin', SequenceType.TIMEOUT: 'timeout',
    SequenceType.RESET: 'timeout', SequenceType.BLOOM: 'bloom', SequenceType.FADE: 'fade'}
# the stages whose message carries the room's eased value too: the white as the room draws it (FlashStage, RevealStage)
FLASH_STAGES: frozenset[str] = frozenset({'flash_in', 'flash_out'})

# the eye's status line as the phones see it (the installation's own language)
LOOKING: str = 'looking for faces'
INSIDE: str = 'looking at someone inside'
WARMING: str = 'warming up'
BUSY: str = 'busy inside'

LinkFactory = Callable[[str, str, Callable[[dict], None], Callable[[bytes], None]], Link]
Caption = Callable[[], tuple[str, str]]     # the credit: name and the visitor's line
Music = Callable[[], tuple[str, str | None]]    # the visitor's track: its id and its file (None without)
Laughs = Callable[[], list[tuple[str, str]]]    # the laughter samples: file name and path
Switch = Callable[[], bool]                     # the panel's laughter switch
Pick = Callable[[], None]                       # Main.pick_style: the visitor's style and track, now


class Phase(Enum):
    OFF = auto()
    IDLE = auto()
    ASKING = auto()
    JUDGING = auto()
    TRANSFORMING = auto()


@dataclass
class Delivery:
    """A turn's result on its way to the phone: made at the flash, the caption read at the capture, the
    saver's files uploaded as soon as they come (the phone fetches the video while the room still shows
    the result), the phone told once."""
    phone: str
    name: str = ''
    line: str = ''
    captioned: bool = False
    saved: tuple[str, bool, str, str, str] | None = None    # id, kept, grin path, portrait path, share path
    over: float | None = None                           # when the turn ended, None while it runs
    started: bool = False


def face_message(frame: FaceFrame, charge: float) -> dict:
    return {'type': 'face', 'charge': round(charge, 3), 'faces': [
        {'box': [round(v, 4) for v in f.crop], 'distance': f.distance, 'subject': f.subject,
         'reject': rejection_label(f) if f.rejection is not None else None} for f in frame.faces]}


class Online:
    def __init__(self, settings: OnlineSettings, camera: Camera, sequencer: Sequencer, face: FaceFinder,
                 grab: ScreenGrab, caption: Caption, music: Music, laughs: Laughs = lambda: [], laughter: Switch = lambda: True,
                 pick: Pick = lambda: None, link_factory: LinkFactory = Link) -> None:
        self.settings: OnlineSettings = settings
        self.camera: Camera = camera
        self.sequencer: Sequencer = sequencer
        self.face: FaceFinder = face
        self.grab: ScreenGrab = grab
        self._caption: Caption = caption
        self._music: Music = music
        self._laughs: Laughs = laughs
        self._laughter: Switch = laughter
        self._pick: Pick = pick
        self._uploading: set[str] = set()   # tracks on their way to the relay
        self._tracks: set[str] = set()      # the tracks the relay has
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
        self._eye_last: tuple[bool, str, bool] | None = None
        self._stage_sent: tuple[str, float] | None = None   # the last stage message: name and progress
        self._stage_time: float = 0.0
        self._phone: str = ''           # the phone of the turn, from the relay's answer
        self._flash_at: float = 0.0     # when the turn's flash came
        self._music_sent: str = ''      # the track the phone was told this turn
        self._timed_out: bool = False   # the sequence reached TIMEOUT: the RESET after it is its natural end
        self._showing: bool = False     # the turn is on the room's screens: from the judging to the last frame
        self._pass: str = ''            # the relay's pass for the phone page's url, '' until the relay sent one
        self._invite: str = ''          # the relay's invite, the pass without an end, for the panel's link
        self._delivery: Delivery | None = None
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
        return self._phase in (Phase.JUDGING, Phase.TRANSFORMING)

    @property
    def showing(self) -> bool:
        """The turn is on the room's screens (the online cell of the walls): from the judging to the end of the
        result, the phone's last frame; the bloom and the fade after it are the room's."""
        return self._showing

    @property
    def pass_value(self) -> str:
        """The relay's current pass, which the QR code's url carries (`?pass=`): the page works for the
        relay's pass lifetime after the scan. '' until the relay sent one: no code without it."""
        return self._pass

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
            self._note(turn=str(message.get('id') or ''))
        elif kind == 'gone':
            self._note(gone=True)
        elif kind == 'pass':
            self._note(pass_value=str(message.get('value') or ''))
        elif kind == 'invite':
            self._note(invite=str(message.get('value') or ''))
        elif kind == 'library':
            self._note(library=(set(str(t) for t in message.get('tracks', [])), set(str(t) for t in message.get('laughs', []))))

    def _on_frame(self, jpeg: bytes) -> None:
        if self._phase != Phase.JUDGING:
            return
        image: np.ndarray | None = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            self.camera.process_online_frame(image)

    def on_face(self, frame: FaceFrame) -> None:
        """Every detector frame (Main.face_callback): the judging goes to the phone during its turn."""
        if self._phase == Phase.JUDGING and self._link is not None:
            self._link.submit(face_message(frame, self._charge))

    def on_saved(self, id: str, style: str, kept: bool, grin: str, portrait: str, share: str = '') -> None:
        """The saver closed the visitor's files (ClipSaver's callback, on its thread)."""
        self._note(saved=(id, style, kept, grin, portrait, share))

    # THE APP CLOCK
    def update(self, now: float, message: SequenceMessage, ready: bool, room_busy: bool) -> None:
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
        if 'pass_value' in notes:
            self._pass = notes['pass_value']
        if 'invite' in notes:
            self._invite = notes['invite']
        if 'saved' in notes and self._delivery is not None:
            id, style, kept, grin, portrait, shared = notes['saved']
            self._delivery.saved = (id, kept, grin, portrait, shared)

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
        # the invite link for the panel, and the button that replaces it
        invite_link: str = f'{http_base(str(s.relay))}/?pass={self._invite}' if connected and self._invite else ''
        if s.invite_link != invite_link:
            s.invite_link = invite_link
        if s.new_invite:
            s.new_invite = False
            if connected:
                link.submit({'type': 'new_invite'})
        if self._phase == Phase.OFF:
            self._phase = Phase.IDLE
        if 'library' in notes:
            self._tracks, laughs = notes['library']
            self._upload_laughs(laughs, link)
        self._deliver(now, link)
        if 'track' in notes:
            # a track just uploaded: the phone of the turn gets it (that far in after the flash), or the line
            id, turn = notes['track']
            self._tracks.add(id)
            self._uploading.discard(id)
            if turn == s.turns and self.turn and id != self._music_sent:
                self._music_sent = id
                self._send_music(link, id, self._music_offset(now))

        # the eye, as the phones see it
        free: bool = self._eye_free(now, seq, ready, room_busy, connected)
        self._send_eye(now, link, free, self._eye_status(seq, ready, room_busy), bool(self._laughter()))

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
                link.submit({'type': 'turn_start', 'window': s.turn_window})
                self._asked = now
                self._phase = Phase.ASKING
        elif self._phase == Phase.ASKING:
            if 'turn' in notes:
                if notes['turn']:
                    self._begin_turn(now, notes['turn'])
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
                # the flash: the capture is taken from the phone's frames, the transformation is the machine's,
                # and the phone sees the screen from here to the end of the timeout
                link.submit({'type': 'stage', 'stage': 'flash', 'in': round(float(self.sequencer.settings.flash_in), 2),
                           'out': round(float(self.sequencer.settings.flash_out), 2)})
                self._phase = Phase.TRANSFORMING
                self._timed_out = False
                self._stage_sent = None
                self._flash_at = now
                self._set_stream()
                self._start_music(now, link)
                self._send_stage(now, link, message)
                if self._delivery is not None and not self._delivery.started:
                    logger.warning('online: the result of the previous turn was never delivered')
                self._delivery = Delivery(self._phone)
            elif seq == SequenceType.CAMERA_SHOW and now - self._turn_start > s.turn_window:
                self._end_turn('missed')
        elif self._phase == Phase.TRANSFORMING:
            s.status = 'turn: transforming'
            if self.camera.online and seq >= SequenceType.ASK:
                # the capture is taken: the room's camera is back underneath, unseen until the result fades
                self.camera.set_online(False)
            frame, last = self.grab.take()
            if frame is not None:
                link.set_frame(frame, last)
                if last:
                    self.grab.set(False)     # the picture stands on the phone from here; the bloom and the fade are the room's
            self._send_stage(now, link, message)
            if self._delivery is not None and not self._delivery.captioned and seq >= SequenceType.ASK:
                # the capture is taken: the visitor's line is drawn, the caption is theirs
                self._delivery.name, self._delivery.line = self._caption()
                self._delivery.captioned = True
            if seq == SequenceType.TIMEOUT and not self._timed_out:
                # the result stands: a few frames a second are enough, the phone is downloading its clip
                self._timed_out = True
                self._set_stream()
            if seq == SequenceType.RESET and self._timed_out and self._seq != SequenceType.RESET:
                # the result has stood its time: the phone's last frame, without the caption (the page fades the
                # caption out itself and keeps the frame; the bloom would brighten it)
                self.grab.last()
                self._showing = False
            if seq == SequenceType.FADE and self._seq != SequenceType.FADE:
                # the fade shows the room's camera: a last frame that never came stops here
                self.grab.set(False)
            if seq == SequenceType.RESET and not self._timed_out:
                # the operator's reset (the sequencer passes RESET at the end of the timeout too: not this)
                self._delivery = None
                self._end_turn('reset')
            elif seq in CAMERA_STATES:
                # the sequence is over, after the bloom and the fade: the turn ends (the files may be on
                # their way already, or follow when the saver has closed them)
                if self._delivery is not None:
                    self._delivery.over = now
                self._end_turn('done')
        self._seq = seq

    def _set_stream(self) -> None:
        """The grab at the panel's width and rate; while the result stands (the timeout) a few frames a second
        at most, the phone is downloading its clip then."""
        s: OnlineSettings = self.settings
        fps: float = float(s.stream_fps)
        if self._timed_out:
            fps = min(fps, STILL_FPS)
        self.grab.set(True, int(s.stream_width), max(1.0, fps))

    def _send_stage(self, now: float, link: Link, message: SequenceMessage) -> None:
        """The stage and its progress to the phone: at every change of stage, else a few times a second
        while the progress moves (the phone's tape follows them as the Max patch follows the room's). The
        flash stages carry the room's eased value too, the white as the room draws it."""
        name: str | None = STAGE_NAMES.get(message.sequence_type)
        if name is None:
            return
        progress: float = round(float(message.progress), 3)
        if self._stage_sent is not None and self._stage_sent[0] == name:
            if self._stage_sent[1] == progress or now - self._stage_time < STAGE_INTERVAL:
                return
        self._stage_sent = (name, progress)
        self._stage_time = now
        stage: dict = {'type': 'stage', 'stage': name, 'progress': progress}
        if name in FLASH_STAGES:
            stage['value'] = round(float(message.value_1), 3)
        link.submit(stage)

    # THE MUSIC
    def _start_music(self, now: float, link: Link) -> None:
        """The visitor's track to the phone, once per turn: at the turn's start (and again at the flash, for a
        pick that changed), at once when the relay has it, else after its upload (the note `track`); the phone
        fetches it during the judging and starts it at the flash, or that far in when it comes later."""
        if not self.settings.music:
            return
        id, path = self._music()
        if not id or id == 'none' or id == self._music_sent:
            return
        if self._ensure_track(link, id, path):
            self._music_sent = id
            self._send_music(link, id, self._music_offset(now))

    def _ensure_track(self, link: Link, id: str, path: str | None) -> bool:
        """True when the relay has the track; else its upload is started (once), and the note `track` follows."""
        if id in self._tracks:
            return True
        if id in self._uploading:
            return False
        if path is None:
            logger.warning('online: no file for track %s', id)
            return False
        self._uploading.add(id)
        turn: int = self.settings.turns
        link.submit_upload([('track', id, path)], lambda ok: self._note(track=(id, turn)) if ok else None)
        return False

    def _music_offset(self, now: float) -> float:
        """How far into the track the room is: the seconds since the flash, 0 before it."""
        return max(0.0, now - self._flash_at) if self._phase == Phase.TRANSFORMING else 0.0

    @staticmethod
    def _send_music(link: Link, id: str, offset: float) -> None:
        link.submit({'type': 'music', 'id': id, 'url': f'/files/tracks/{id}.m4a', 'offset': round(max(0.0, offset), 2)})

    def _upload_laughs(self, present: set[str], link: Link) -> None:
        """The laughter samples the relay lacks, in one upload (once per relay; 8 MB in all)."""
        missing: list[tuple[str, str, str]] = [('laugh', os.path.splitext(name)[0], path) for name, path in self._laughs() if name not in present]
        if not missing:
            return
        logger.info('online: uploading %d laughter samples to the relay', len(missing))
        link.submit_upload(missing, lambda ok: None if ok else logger.warning('online: not every laughter sample reached the relay'))

    def _deliver(self, now: float, link: Link) -> None:
        """The result: upload the saved files as soon as the saver has closed them (during the timeout, so
        the phone has the video when the turn ends) and tell the phone, once."""
        delivery: Delivery | None = self._delivery
        if delivery is None or delivery.started:
            return
        result: dict = {'type': 'result', 'phone': delivery.phone, 'name': delivery.name, 'caption': delivery.line}
        if delivery.saved is not None:
            id, kept, grin, portrait, shared = delivery.saved
            delivery.started = True
            result['id'] = id
            if kept:
                # the phone's video is the share clip (the track on the tape, the watermark); the walls' clip without one
                files: list[tuple[str, str, str]] = [('portrait', id, portrait), ('grin', id, shared or grin)]
                link.submit_upload(files, lambda ok: link.submit({**result, 'ok': ok}))
            else:
                logger.warning('online: the visitor %s was not saved, no result', id)
                link.submit({**result, 'ok': False})
        elif delivery.over is not None and now - delivery.over > SAVE_TIMEOUT:
            delivery.started = True
            logger.warning('online: no saved files within %.0f s of the turn, no result', SAVE_TIMEOUT)
            link.submit({**result, 'id': '', 'ok': False})

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

    def _send_eye(self, now: float, link: Link, free: bool, status: str, laughter: bool) -> None:
        if not link.connected:
            self._eye_last = None
            return
        eye: tuple[bool, str, bool] = (free, status, laughter)
        if eye != self._eye_last or now - self._eye_sent > EYE_INTERVAL:
            self._eye_last = eye
            self._eye_sent = now
            link.submit({'type': 'eye', 'free': free, 'status': status, 'laughter': laughter})

    # THE TURN
    def _begin_turn(self, now: float, phone: str) -> None:
        self._turn_start = now
        self._phone = phone
        self._music_sent = ''
        self._phase = Phase.JUDGING
        self._showing = True
        self._charge = 0.0
        self.face.reset()               # the room's last faces are not this visitor's
        self.camera.set_online(True)
        self.settings.turns += 1
        self._pick()                    # the style and the track now: the phone fetches the track during the judging
        if self._link is not None:
            self._start_music(now, self._link)
        logger.info('online: turn %d starts', self.settings.turns)

    def _end_turn(self, reason: str | None) -> None:
        """Back to the room: the camera, and the phone is told why (None: nothing can be sent)."""
        if not self.turn and self._phase != Phase.ASKING:
            return
        self._showing = False
        self.camera.set_online(False)
        self.grab.set(False)
        if reason is not None and self._link is not None:
            self._link.submit({'type': 'turn_end', 'reason': reason})
        if self.turn:
            logger.info('online: turn ends (%s)', reason)
        self._phase = Phase.IDLE

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
        self.settings.invite_link = ''
        self._eye_last = None
        self._pass = ''
        self._invite = ''
