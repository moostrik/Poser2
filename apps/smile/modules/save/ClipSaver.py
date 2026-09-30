import cv2
import numpy as np
import threading
import queue
import os
import datetime
import logging

from modules.render.settings import SequencerSettings
from modules.video import ffmpeg
from modules.video.ffmpeg import ClipWriter, Encoder

logger = logging.getLogger(__name__)

Command = tuple


class ClipSaver:
    """Saves each visitor for the players: the grin as an h264 mp4, the capture and the portrait as stills.

    Files are built in recordings/temp/ and moved (an atomic rename) to recordings/grins/ and
    recordings/portraits/ when the sequence ends complete, else deleted. The save thread is the only
    owner of the encoder process; the other threads only enqueue commands, in order."""

    def __init__(self, settings: SequencerSettings, root: str = 'recordings/', max_queue_size: int = 64) -> None:
        self.settings: SequencerSettings = settings
        self.temp_dir: str = os.path.join(root, 'temp')
        self.grins_dir: str = os.path.join(root, 'grins')
        self.portraits_dir: str = os.path.join(root, 'portraits')
        for folder in (self.temp_dir, self.grins_dir, self.portraits_dir):
            os.makedirs(folder, exist_ok=True)
        self._clean_temp()

        self.enabled: bool = ffmpeg.available()
        if not self.enabled:
            logger.error('ffmpeg not found on the PATH: visitors are not saved')
        self.encoder: Encoder = ffmpeg.pick_encoder(settings.encoder) if self.enabled else Encoder.CPU
        logger.info('Clip encoder: %s', self.encoder.value)

        self.sequence_count: int = 0
        self.queue: queue.Queue[Command] = queue.Queue(max_queue_size)
        self._dropped: bool = False
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._save_loop, name='ClipSaver')
        self.thread.start()
        self.settings.bind(SequencerSettings.save, self._on_save)

    # ANY THREAD

    def new_sequence(self, capture: np.ndarray) -> None:
        """Open the visitor's files and save the capture; the fps is read here, not on the save thread."""
        if not self._active():
            return
        self.sequence_count += 1
        id: str = self._create_sequence_id(self.sequence_count)
        self.queue.put(('open', id, float(self.settings.grin_fps)))
        self.queue.put(('capture', capture))

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
                        self._finish(id, writer, keep=False)
                    id = command[1]
                    writer = ClipWriter(self._temp_path(id, '.mp4'), command[2], self.encoder)
                elif writer is None:
                    continue
                elif kind == 'capture':
                    cv2.imwrite(self._temp_path(id, '_capture.jpg'), command[1])
                elif kind == 'portrait':
                    cv2.imwrite(self._temp_path(id, '_portrait.jpg'), command[1])
                elif kind == 'frame':
                    writer.add(command[1])
                elif kind == 'close':
                    self._finish(id, writer, keep=command[1])
                    writer = None
            except Exception as e:
                logger.error('Error saving %s: %s', id, e)
                if writer is not None:
                    writer.failed = True
        if writer is not None:
            self._finish(id, writer, keep=False)

    def _finish(self, id: str, writer: ClipWriter, keep: bool) -> None:
        complete: bool = writer.close()
        temp = {name: self._temp_path(id, name) for name in ('.mp4', '_capture.jpg', '_portrait.jpg')}
        if keep and complete:
            os.replace(temp['.mp4'], os.path.join(self.grins_dir, id + '.mp4'))
            for name in ('_capture.jpg', '_portrait.jpg'):
                if os.path.exists(temp[name]):
                    os.replace(temp[name], os.path.join(self.portraits_dir, id + name))
            logger.info('Saved %s (%d frames)', id, writer.frames)
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
    def _create_sequence_id(count: int) -> str:
        timestamp: str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        return timestamp + ("_A" if count % 2 == 0 else "_B")
