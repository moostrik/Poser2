import cv2
import numpy as np
import threading
import queue
import os
import datetime
import json
import logging
from typing import Callable, Protocol

from . import share
from ..paths import OUTPUT
from .record import embed_jpeg
from modules import video
from modules.settings import BaseSettings, Field
from modules.utils import Broadcast

logger = logging.getLogger(__name__)

Command = tuple
SavedCallback = Callable[[str, str, bool, str, str, str], None]   # id, style, kept, grin path, portrait path, share path

STYLES: tuple[str, ...] = ('A', 'B')


class ClipSaverSettings(BaseSettings):
    """Whether each finished visitor is saved, and the chip its grin clip is encoded on."""
    enabled: Field[bool]           = Field(True, pinned=True, label="Save visitors", description="Save each finished visitor's grin, capture and portrait to output/")
    encoder: Field[video.Hardware] = Field(video.Hardware.AUTO, access=Field.INIT, label="Encoder", description="The chip the grin clip is encoded on (Auto tries the iGPU, the GPU, the CPU)")


class GrinRate(Protocol):
    """What the saver reads of the grin's playback settings: the speed its clip is written at (the sequence's
    `GrinPlayerSettings`, which the saver cannot import: the sequence's settings hold the saver's)."""
    fps: Field[float]


class ClipSaver:
    """Saves each visitor for the clip player: the grin as an h264 mp4, the capture and the portrait as stills.

    Files are built in output/temp/ and moved (an atomic rename) to output/grins/<style>/ and
    output/portraits/<style>/ when the sequence ends complete, else deleted. The clip player reads the
    style (A or B) from that folder. The clip and the portrait carry the visitor's record: how they
    were made (`save/record.py`). After the clip, the share clip for the visitor's phone
    (`save/share.py`, `shares/<style>/<id>.mp4`): the same grin with the visitor's track as its
    sound (`submit_share`: the file and the point of it the room hears when the grin starts) on the tape, and
    the watermark; the clip player's clip stays silent. The share clip is built first, the clip player's clip
    lands after it: the visitor outside is waiting for theirs. The save thread is the only
    owner of the encoder process; the other threads only enqueue commands, in order."""

    def __init__(self, settings: ClipSaverSettings, grin: GrinRate, credit_name: Callable[[], str], root: str = OUTPUT,
                 max_queue_size: int = 64) -> None:
        self.settings: ClipSaverSettings = settings
        self._grin: GrinRate = grin             # the clip's fps, read at each visitor's open
        self._credit_name: Callable[[], str] = credit_name     # the watermark's name, read at each share
        self.temp_dir: str = os.path.join(root, 'temp')
        self.grins_dir: str = os.path.join(root, 'grins')
        self.portraits_dir: str = os.path.join(root, 'portraits')
        self.shares_dir: str = os.path.join(root, 'shares')
        self._watermark_path: str = os.path.join(self.shares_dir, 'watermark.png')     # beside the share clips
        self._watermark_name: str | None = None     # the name the watermark file holds; None before it is written

        self.available: bool = False        # ffmpeg is there: known at the start
        self.encoder: video.Hardware = video.Hardware.CPU

        self.queue: queue.Queue[Command] = queue.Queue(max_queue_size)
        self._share: tuple[str | None, float] | None = None   # the open visitor's track and offset (save thread)
        self._dropped: bool = False
        self._saved_callbacks: Broadcast[[str, str, bool, str, str, str]] = Broadcast(name='saved')
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._save_loop, name='ClipSaver')
        self.settings.bind(ClipSaverSettings.enabled, self._on_save)

    def start(self) -> None:
        """The folders made, what a crash left in temp/ cleared, the encoder picked, and the save thread started."""
        os.makedirs(self.temp_dir, exist_ok=True)
        for style in STYLES:
            os.makedirs(os.path.join(self.grins_dir, style), exist_ok=True)
            os.makedirs(os.path.join(self.portraits_dir, style), exist_ok=True)
            os.makedirs(os.path.join(self.shares_dir, style), exist_ok=True)
        self._clean_temp()
        self.available = video.available()
        if not self.available:
            logger.error('ffmpeg not found on the PATH: visitors are not saved')
        self.encoder = video.pick_encoder(self.settings.encoder) if self.available else video.Hardware.CPU
        logger.info('Clip encoder: %s', self.encoder.value)
        self.thread.start()

    # ANY THREAD

    def submit_open(self, capture: np.ndarray, style: str) -> None:
        """Open the visitor's files and save the capture. style is the visitor's style (A or B): the
        folder its files end up in. The fps is read here, not on the save thread."""
        if not self._active():
            return
        id: str = self._create_sequence_id()
        self.queue.put(('open', id, float(self._grin.fps), style or STYLES[0]))
        self.queue.put(('capture', capture))

    def submit_record(self, record: dict) -> None:
        """How the visitor is made (`save/record.py`): goes into the portrait and the clip, so
        hand it over before the portrait."""
        if self._active():
            self.queue.put(('record', record))

    def submit_portrait(self, image: np.ndarray) -> None:
        if self._active():
            self.queue.put(('portrait', image))

    def submit_share(self, track: str | None, offset: float) -> None:
        """The share clip's sound: the track's file and the second of it that plays when the grin starts in
        the room (None: silence)."""
        if self._active():
            self.queue.put(('share', track, offset))

    def submit_frame(self, image: np.ndarray) -> None:
        """A grin frame; dropped with a warning when the encoder is that far behind, so the caller never blocks."""
        if not self._active():
            return
        try:
            self.queue.put_nowait(('frame', image))
            self._dropped = False
        except queue.Full:
            if not self._dropped:
                logger.warning('Save queue full, dropping grin frames')
                self._dropped = True

    def submit_close(self, keep: bool) -> None:
        """Close the visitor's files: kept for the clip player when the sequence is complete, else deleted."""
        if self._active():
            self.queue.put(('close', keep))

    def stop(self) -> None:
        self.settings.unbind(ClipSaverSettings.enabled, self._on_save)
        if self.thread.ident is None:       # never started
            return
        self.queue.put(('close', False))
        self.stop_event.set()
        self.thread.join()

    def _active(self) -> bool:
        return self.available and bool(self.settings.enabled)

    def _on_save(self, value: bool) -> None:
        # switching off drops the unfinished visitor; switching on waits for the next capture
        if not value:
            self.queue.put(('close', False))

    # SAVE THREAD

    def _save_loop(self) -> None:
        id: str = ''
        style: str = STYLES[0]
        writer: video.ClipWriter | None = None
        while not (self.stop_event.is_set() and self.queue.empty()):
            try:
                command = self.queue.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                kind = command[0]
                if kind == 'open':
                    if writer is not None:
                        self._finish(id, style, writer, keep=False)
                    id = command[1]
                    style = command[3]
                    writer = video.ClipWriter(self._temp_path(id, '.mp4'), command[2], video.h264_encode(self.encoder))
                    self._share = None
                elif writer is None:
                    continue
                elif kind == 'capture':
                    cv2.imwrite(self._temp_path(id, '_capture.jpg'), command[1])
                elif kind == 'record':
                    # the encoder starts with the first grin frame, so the clip still takes it
                    writer.comment = json.dumps({'id': id, 'style': style, **command[1]}, separators=(',', ':'))
                elif kind == 'portrait':
                    ok, jpeg = cv2.imencode('.jpg', command[1])
                    if not ok:
                        raise ValueError('could not encode the portrait')
                    data: bytes = jpeg.tobytes()
                    if writer.comment:
                        data = embed_jpeg(data, writer.comment)
                    with open(self._temp_path(id, '_portrait.jpg'), 'wb') as f:
                        f.write(data)
                elif kind == 'frame':
                    writer.add(command[1])
                elif kind == 'share':
                    self._share = (command[1] or None, float(command[2]))
                elif kind == 'close':
                    self._finish(id, style, writer, keep=command[1])
                    writer = None
            except Exception as e:
                logger.error('Error saving %s: %s', id, e)
                if writer is not None:
                    writer.failed = True
        if writer is not None:
            self._finish(id, style, writer, keep=False)

    def _finish(self, id: str, style: str, writer: video.ClipWriter, keep: bool) -> None:
        complete: bool = writer.close()
        temp = {name: self._temp_path(id, name) for name in ('.mp4', '_capture.jpg', '_portrait.jpg')}
        kept: bool = keep and complete
        to_share, self._share = self._share, None
        grin: str = ''
        portrait: str = ''
        shared: str = ''
        if kept:
            grins: str = os.path.join(self.grins_dir, style)
            portraits: str = os.path.join(self.portraits_dir, style)
            os.makedirs(grins, exist_ok=True)
            os.makedirs(portraits, exist_ok=True)
            # the stills first: the clip landing is what the clip player reacts to
            for name in ('_capture.jpg', '_portrait.jpg'):
                if os.path.exists(temp[name]):
                    os.replace(temp[name], os.path.join(portraits, id + name))
            grin = os.path.join(grins, id + '.mp4')
            portrait = os.path.join(portraits, id + '_portrait.jpg')
            # the phone's clip first, from the finished file where it is: the visitor outside waits for it,
            # the wall only gets one more clip
            shared = self._build_share(id, style, temp['.mp4'], to_share)
            os.replace(temp['.mp4'], grin)
            logger.info('Saved %s %s (%d frames)', style, id, writer.frames)
        else:
            for path in temp.values():
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
        self._saved_callbacks(id, style, kept, grin, portrait, shared)

    def _watermark(self) -> str | None:
        """The watermark file with the credit name as it is now (save thread): written again only when the name
        differs from the file's; None without a name, or when it could not be written."""
        name: str = self._credit_name()
        if not name:
            return None
        if name != self._watermark_name:
            if not share.write_watermark(name, self._watermark_path):
                return None
            self._watermark_name = name
        return self._watermark_path

    def _build_share(self, id: str, style: str, grin: str, to_share: tuple[str | None, float] | None) -> str:
        """The share clip from the landed grin: its path, or '' when it could not be made (a warning)."""
        temp: str = self._temp_path(id, '_share.mp4')
        path: str = os.path.join(self.shares_dir, style, id + '.mp4')
        track, offset = to_share if to_share is not None else (None, 0.0)
        try:
            share.build_share(grin, track, offset, self._watermark(), self.encoder, temp)
            os.replace(temp, path)
            return path
        except Exception as e:
            logger.warning('No share clip for %s: %s', id, e)
            try:
                os.remove(temp)
            except FileNotFoundError:
                pass
            return ''

    def add_saved_callback(self, callback: SavedCallback) -> None:
        """Called on the save thread when a visitor's files are closed: (id, style, kept, grin path,
        portrait path, share path); the paths are empty when the files were not kept (the share path
        also when it could not be made)."""
        self._saved_callbacks.add_callback(callback)

    def _temp_path(self, id: str, suffix: str) -> str:
        return os.path.join(self.temp_dir, id + suffix)

    def _clean_temp(self) -> None:
        # a crash leaves unfinished files behind
        for name in os.listdir(self.temp_dir):
            try:
                os.remove(os.path.join(self.temp_dir, name))
            except OSError as e:
                logger.warning('Could not remove %s: %s', name, e)

    @staticmethod
    def _create_sequence_id() -> str:
        return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
