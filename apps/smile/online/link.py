"""The connection to the relay (relay/server.py): a websocket client on its own thread that reconnects
by itself, hands every message to a callback and sends what the other threads queue. The screen frames
for the phone (`send_frame`) have one slot, the newest wins, and are JPEG-encoded on the link's thread,
never on the render thread. Uploads (the visitor's files, over HTTP) run on a third thread, so a slow
upload never holds the frames.

Nothing here ends the thread: a relay that is down, refuses the token or sends rubbish is a warning
(not at every try) and the next try. Messages are JSON text; images are binary with one leading byte
(relay.server: FRAME 0 from the phone, SHOW 1 to the phone, LAST 2 the turn's last frame)."""

import json
import logging
import os
import queue
import ssl
import tempfile
import time
from threading import Event, Lock, Thread
from typing import Any, Callable

import cv2
import numpy as np
import requests
import websocket

from modules import video

logger = logging.getLogger(__name__)

FRAME: int = 0
SHOW: int = 1
LAST: int = 2                       # the turn's last screen frame: the phone fades to it and keeps it
RECONNECT_INTERVAL: float = 3.0
RECEIVE_TIMEOUT: float = 0.05       # s; between receives the queued messages go out
WARN_INTERVAL: float = 60.0         # s between the log lines of a connection that keeps failing
UPLOAD_TIMEOUT: float = 60.0
STREAM_QUALITY: int = 70            # JPEG, the screen frames for the phone

MessageCallback = Callable[[dict], None]
FrameCallback = Callable[[bytes], None]
DoneCallback = Callable[[bool], None]


def read_token(path: str) -> str:
    try:
        with open(path, encoding='utf-8') as f:
            return f.read().strip()
    except OSError:
        return ''


def encode(image: np.ndarray, quality: int = STREAM_QUALITY) -> bytes | None:
    """A JPEG of a BGR image for the phone's window."""
    ok, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpeg.tobytes() if ok else None


def http_base(ws_url: str) -> str:
    """The relay's HTTP address from its websocket address: wss://host/ws/smile -> https://host."""
    scheme, _, rest = ws_url.partition('://')
    host: str = rest.split('/', 1)[0]
    return ('https' if scheme == 'wss' else 'http') + '://' + host


class Link(Thread):
    def __init__(self, url: str, token: str, on_message: MessageCallback, on_frame: FrameCallback, name: str = 'online-link') -> None:
        super().__init__(name=name, daemon=True)
        self.url: str = url
        self.token: str = token
        self._on_message: MessageCallback = on_message
        self._on_frame: FrameCallback = on_frame
        self._out: queue.Queue[str | bytes] = queue.Queue(maxsize=256)
        self._uploads: queue.Queue[tuple[list[tuple[str, str, str]], DoneCallback]] = queue.Queue()
        self._frame: tuple[np.ndarray, bool] | None = None      # the newest screen frame (RGBA, last) not sent yet
        self._frame_mutex: Lock = Lock()
        self._stop: Event = Event()
        self._ws: websocket.WebSocket | None = None
        self._connected: bool = False
        self._warned: float = 0.0
        self._uploader: Thread = Thread(target=self._upload_loop, name=name + '-upload', daemon=True)

    @property
    def connected(self) -> bool:
        return self._connected

    def stop(self) -> None:
        """Ends the threads; does not wait for them (called from the render thread when the panel's
        switch goes off), they are daemons and stop at their next step."""
        self._stop.set()
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    # SENDING, from any thread
    def send(self, message: dict) -> None:
        self._queue(json.dumps(message, separators=(',', ':')))

    def send_image(self, jpeg: bytes) -> None:
        """An image for the phone's window (SHOW), encoded already."""
        self._queue(bytes([SHOW]) + jpeg)

    def send_frame(self, frame: np.ndarray, last: bool = False) -> None:
        """A screen frame (RGBA, the render's grab) for the phone's window: replaces one not sent yet;
        `last` is the turn's last, the one the phone keeps."""
        with self._frame_mutex:
            self._frame = (frame, last)

    def _queue(self, item: str | bytes) -> None:
        try:
            self._out.put_nowait(item)
        except queue.Full:
            logger.debug('online link: outgoing queue full, dropping')

    def upload(self, files: list[tuple[str, str, str]], done: DoneCallback) -> None:
        """Upload files [(kind, id, path)] over HTTP, then call done(all_ok) on the upload thread."""
        self._uploads.put((files, done))

    # THE THREADS
    def run(self) -> None:
        self._uploader.start()
        while not self._stop.is_set():
            try:
                self._connect_and_serve()
            except Exception as e:
                self._warn('online link: %s', e)
            finally:
                self._disconnect()
            self._stop.wait(RECONNECT_INTERVAL)

    def _connect_and_serve(self) -> None:
        ws = websocket.WebSocket()
        ws.connect(self.url, header=[f'Authorization: Bearer {self.token}'], timeout=5)
        self._ws = ws
        self._connected = True
        self._warned = 0.0
        logger.info('online link: connected to %s', self.url)
        self._on_message({'type': 'library', **self._library()})
        while not self._stop.is_set():
            self._flush(ws)
            ws.settimeout(RECEIVE_TIMEOUT)
            try:
                message: Any = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if isinstance(message, bytes):
                if message and message[0] == FRAME:
                    self._on_frame(message[1:])
            elif isinstance(message, str) and message:
                try:
                    self._on_message(json.loads(message))
                except Exception:
                    logger.exception('online link: a message failed')
            else:
                break   # closed

    def _flush(self, ws: websocket.WebSocket) -> None:
        while True:
            try:
                item: str | bytes = self._out.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, bytes):
                ws.send_binary(item)
            else:
                ws.send(item)
        with self._frame_mutex:
            item, self._frame = self._frame, None
        if item is not None:
            frame, last = item
            jpeg: bytes | None = encode(cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR))
            if jpeg is not None:
                ws.send_binary(bytes([LAST if last else SHOW]) + jpeg)

    @staticmethod
    def _track_aac(path: str) -> bytes:
        """A music track as AAC for the phones (every phone decodes it; Safari refuses some MP3s)."""
        fd, temp = tempfile.mkstemp(suffix='.m4a')
        os.close(fd)
        try:
            video.transcode_aac(path, temp)
            with open(temp, 'rb') as f:
                return f.read()
        finally:
            try:
                os.remove(temp)
            except OSError:
                pass

    def _library(self) -> dict:
        """The music tracks and laughter samples the relay has (its /api/library), so only the missing
        ones are uploaded: {'tracks': [ids], 'laughs': [file names]}."""
        try:
            response = requests.get(f'{http_base(self.url)}/api/library', headers={'Authorization': f'Bearer {self.token}'}, timeout=5)
            response.raise_for_status()
            library: dict = response.json()
            return {'tracks': [str(t) for t in library.get('tracks', [])], 'laughs': [str(t) for t in library.get('laughs', [])]}
        except Exception as e:
            logger.warning("online link: the relay's library: %s", e)
            return {'tracks': [], 'laughs': []}

    def _disconnect(self) -> None:
        if self._connected:
            logger.info('online link: disconnected')
        self._connected = False
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        # what was queued for a connection that is gone is stale
        with self._frame_mutex:
            self._frame = None
        while not self._out.empty():
            try:
                self._out.get_nowait()
            except queue.Empty:
                break

    def _warn(self, text: str, *args) -> None:
        now: float = time.time()
        if now - self._warned > WARN_INTERVAL:
            self._warned = now
            logger.warning(text, *args)

    def _upload_loop(self) -> None:
        base: str = http_base(self.url)
        while not self._stop.is_set():
            try:
                files, done = self._uploads.get(timeout=0.2)
            except queue.Empty:
                continue
            ok: bool = True
            for kind, id, path in files:
                try:
                    data: bytes = self._track_aac(path) if kind == 'track' else open(path, 'rb').read()
                    ext: str = 'm4a' if kind == 'track' else os.path.splitext(path)[1].lstrip('.').lower()
                    response = requests.post(f'{base}/api/upload', params={'kind': kind, 'id': id, 'ext': ext},
                                             headers={'Authorization': f'Bearer {self.token}'}, data=data, timeout=UPLOAD_TIMEOUT)
                    response.raise_for_status()
                except Exception as e:
                    ok = False
                    logger.warning('online link: upload of %s %s failed: %s', kind, os.path.basename(path), e)
            try:
                done(ok)
            except Exception:
                logger.exception('online link: the upload callback failed')
