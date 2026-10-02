import cv2
import numpy as np
import threading
import queue
import os
import datetime
import json
import logging

from modules.render.settings import SequencerSettings
from modules.save.record import embed_jpeg
from modules.video import ffmpeg
from modules.video.ffmpeg import ClipWriter, Encoder

logger = logging.getLogger(__name__)

Command = tuple

STYLES: tuple[str, ...] = ('A', 'B')


class ClipSaver:
    """Saves each visitor for the players: the grin as an h264 mp4, the capture and the portrait as stills.

    Files are built in output/temp/ and moved (an atomic rename) to output/grins/<style>/ and
    output/portraits/<style>/ when the sequence ends complete, else deleted. The players read the
    style (A or B) from that folder. The clip and the portrait carry the visitor's record: how they
    were made (modules/save/record.py). The save thread is the only
    owner of the encoder process; the other threads only enqueue commands, in order."""

    def __init__(self, settings: SequencerSettings, root: str = 'output/', max_queue_size: int = 64) -> None:
        self.settings: SequencerSettings = settings
        self.temp_dir: str = os.path.join(root, 'temp')
        self.grins_dir: str = os.path.join(root, 'grins')
        self.portraits_dir: str = os.path.join(root, 'portraits')
        os.makedirs(self.temp_dir, exist_ok=True)
        for style in STYLES:
            os.makedirs(os.path.join(self.grins_dir, style), exist_ok=True)
            os.makedirs(os.path.join(self.portraits_dir, style), exist_ok=True)
        self._clean_temp()

        self.enabled: bool = ffmpeg.available()
        if not self.enabled:
            logger.error('ffmpeg not found on the PATH: visitors are not saved')
        self.encoder: Encoder = ffmpeg.pick_encoder(settings.encoder) if self.enabled else Encoder.CPU
        logger.info('Clip encoder: %s', self.encoder.value)

        self.queue: queue.Queue[Command] = queue.Queue(max_queue_size)
        self._dropped: bool = False
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._save_loop, name='ClipSaver')
        self.thread.start()
        self.settings.bind(SequencerSettings.save, self._on_save)

    # ANY THREAD

    def new_sequence(self, capture: np.ndarray, style: str) -> None:
        """Open the visitor's files and save the capture. style is the visitor's style (A or B): the
        folder its files end up in. The fps is read here, not on the save thread."""
        if not self._active():
            return
        id: str = self._create_sequence_id()
        self.queue.put(('open', id, float(self.settings.grin_fps), style or STYLES[0]))
        self.queue.put(('capture', capture))

    def set_record(self, record: dict) -> None:
        """How the visitor is made (modules/save/record.py): goes into the portrait and the clip, so
        hand it over before the portrait."""
        if self._active():
            self.queue.put(('record', record))

    def save_portrait(self, image: np.ndarray) -> None:
        if self._active():
            self.queue.put(('portrait', image))

    def save_frame(self, image: np.ndarray) -> None:
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

    def end_sequence(self, keep: bool) -> None:
        """Close the visitor's files: kept for the players when the sequence is complete, else deleted."""
        if self._active():
            self.queue.put(('close', keep))

    def stop(self) -> None:
        self.settings.unbind(SequencerSettings.save, self._on_save)
        self.queue.put(('close', False))
        self.stop_event.set()
        self.thread.join()

    def _active(self) -> bool:
        return self.enabled and bool(self.settings.save)

    def _on_save(self, value: bool) -> None:
        # switching off drops the unfinished visitor; switching on waits for the next capture
        if not value:
            self.queue.put(('close', False))

    # SAVE THREAD

    def _save_loop(self) -> None:
        id: str = ''
        style: str = STYLES[0]
        writer: ClipWriter | None = None
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
                    writer = ClipWriter(self._temp_path(id, '.mp4'), command[2], self.encoder)
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
                elif kind == 'close':
                    self._finish(id, style, writer, keep=command[1])
                    writer = None
            except Exception as e:
                logger.error('Error saving %s: %s', id, e)
                if writer is not None:
                    writer.failed = True
        if writer is not None:
            self._finish(id, style, writer, keep=False)

    def _finish(self, id: str, style: str, writer: ClipWriter, keep: bool) -> None:
        complete: bool = writer.close()
        temp = {name: self._temp_path(id, name) for name in ('.mp4', '_capture.jpg', '_portrait.jpg')}
        if keep and complete:
            grins: str = os.path.join(self.grins_dir, style)
            portraits: str = os.path.join(self.portraits_dir, style)
            os.makedirs(grins, exist_ok=True)
            os.makedirs(portraits, exist_ok=True)
            # the stills first: the clip landing is what the players react to
            for name in ('_capture.jpg', '_portrait.jpg'):
                if os.path.exists(temp[name]):
                    os.replace(temp[name], os.path.join(portraits, id + name))
            os.replace(temp['.mp4'], os.path.join(grins, id + '.mp4'))
            logger.info('Saved %s %s (%d frames)', style, id, writer.frames)
        else:
            for path in temp.values():
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass

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
