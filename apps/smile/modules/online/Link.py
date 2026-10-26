"""The connection to the relay (relay/server.py): a websocket client on its own thread that reconnects
by itself, hands every message to a callback and sends what the other threads queue. Uploads (the
visitor's files, over HTTP) run on a second thread, so a slow upload never holds the frames.

Nothing here ends the thread: a relay that is down, refuses the token or sends rubbish is a warning
(not at every try) and the next try. Messages are JSON text; images are binary with one leading byte
(relay.server: FRAME 0 from the phone, SHOW 1 to the phone)."""

import json
import logging
import os
import queue
import ssl
import time
from threading import Event, Thread
from typing import Any, Callable

import requests
import websocket

logger = logging.getLogger(__name__)

FRAME: int = 0
SHOW: int = 1
RECONNECT_INTERVAL: float = 3.0
RECEIVE_TIMEOUT: float = 0.05       # s; between receives the queued messages go out
WARN_INTERVAL: float = 60.0         # s between the log lines of a connection that keeps failing
UPLOAD_TIMEOUT: float = 60.0

MessageCallback = Callable[[dict], None]
FrameCallback = Callable[[bytes], None]
DoneCallback = Callable[[bool], None]


def read_token(path: str) -> str:
    try:
        with open(path, encoding='utf-8') as f:
            return f.read().strip()
    except OSError:
        return ''


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
        """An image for the phone's window (SHOW)."""
        self._queue(bytes([SHOW]) + jpeg)

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
        ws.connect(f'{self.url}?token={self.token}', timeout=5)
        self._ws = ws
        self._connected = True
        self._warned = 0.0
        logger.info('online link: connected to %s', self.url)
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
                return
            if isinstance(item, bytes):
                ws.send_binary(item)
            else:
                ws.send(item)

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
                    with open(path, 'rb') as f:
                        data: bytes = f.read()
                    response = requests.post(f'{base}/api/upload', params={'token': self.token, 'kind': kind, 'id': id},
                                             data=data, timeout=UPLOAD_TIMEOUT)
                    response.raise_for_status()
                except Exception as e:
                    ok = False
                    logger.warning('online link: upload of %s %s failed: %s', kind, os.path.basename(path), e)
            try:
                done(ok)
            except Exception:
                logger.exception('online link: the upload callback failed')
