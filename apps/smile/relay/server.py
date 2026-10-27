"""The relay of SMILE Online: one small server between the phones and SMILE.

    python -m relay.server --port 8800 --token-file relay/token.txt [--data relay/data] [--web relay/web]
                           [--cert cert.pem --key key.pem]

It serves the phone page (`--web`, the files of relay/web), a websocket for every phone (`/ws/phone`),
one for SMILE (`/ws/smile?token=...`), uploads from SMILE (`POST /api/upload?token=&kind=&id=`) and the
visitors' files (`/files/<kind>s/<name>`, kept under `--data`). It keeps the line (relay/line.py) and
passes messages; it decides nothing about a turn: SMILE starts and ends every turn.

Messages are JSON text, images are binary with one leading byte: FRAME (0) a phone's camera frame to
SMILE, SHOW (1) an image from SMILE for the phone's window. The phone with the turn is the only one
whose frames are passed on, and only while SMILE is connected.

Phone -> relay: join, leave; frames.
Relay -> phone: state (see Line.state), grid (the newest stills), turn, face, stage, result, turn_end;
Relay -> SMILE: line {waiting}, turn {id}, gone;
    show images.
phone -> relay: join, leave, seen (a screen frame drawn), timing (its turn's end, for the log);
SMILE -> relay: eye {free, status, laughter}, turn_start, face, stage {flash, in, out} then stage {name, progress}
    through the sequence, music {id, url, offset} (the visitor's track, which SMILE uploaded as kind `track` the
    first time; the laughter samples as kind `laugh` with `ext`; `GET /api/library` lists what the relay has),
    turn_end {reason}, and after the turn
    result {id, phone, name, caption, ok} (to that phone, whatever it does now; without `phone` to the turn's);
    show images; uploads over HTTP.
The screen frames (SHOW) are paced by the phone: one is in flight at a time, the phone answers each with
`seen`, and a newer frame from SMILE replaces one still waiting. So a slow phone gets fewer frames, not
frames seconds late, and the messages behind them (stage, result, turn_end) are never held up in the pipe.
Relay -> SMILE: line {waiting}, turn {id}, gone (the phone with the turn left); frames.

Phones only open their camera on HTTPS: run it behind `tailscale serve`, a reverse proxy, or with
`--cert` and `--key`."""

import argparse
import asyncio
import json
import logging
import logging.handlers
import os
import secrets
import ssl
import time
import sys
import uuid
from collections import deque
from typing import Any

from aiohttp import WSMsgType, web

from relay.line import Line

logger = logging.getLogger('relay')

FRAME: int = 0      # a phone's camera frame, to SMILE
SHOW: int = 1       # an image from SMILE, to the phone with the turn

KINDS: dict[str, str] = {'still': '.jpg', 'portrait': '.jpg', 'grin': '.mp4'}     # a visitor's files, and their folders (kind + 's')
UPLOAD_KINDS: dict[str, str] = {**KINDS, 'track': '.m4a', 'laugh': ''}            # and the library: tracks/ (AAC, for every phone), laughs/ (their own extension)
LAUGH_EXTENSIONS: tuple[str, ...] = ('mp3', 'wav')
MAX_FRAME: int = 512 * 1024             # bytes; a phone frame larger than this is dropped
MAX_UPLOAD: int = 64 * 1024 * 1024
GRID_SIZE: int = 32                     # stills the phones are told about
HEARTBEAT: float = 20.0                 # s; a websocket ping, so a dead phone leaves the line
FRAME_PATIENCE: float = 2.0             # s a screen frame may stay unanswered before the next goes anyway


class Relay:
    def __init__(self, token: str, data: str) -> None:
        self.token: str = token
        self.data: str = data
        self.line: Line = Line()
        self.phones: dict[str, web.WebSocketResponse] = {}
        self.smile: web.WebSocketResponse | None = None
        self.stills: deque[str] = deque(maxlen=GRID_SIZE)      # urls, the newest last
        self._frame_pending: bytes | None = None                # the newest screen frame not sent yet
        self._frame_in_flight: float | None = None              # when the frame the phone has not seen yet was sent
        self._flights: list[float] = []                         # the flight times of the turn's frames, for the log
        for kind, ext in UPLOAD_KINDS.items():
            os.makedirs(os.path.join(data, kind + 's'), exist_ok=True)
        folder: str = os.path.join(data, 'stills')
        for name in sorted(os.listdir(folder))[-GRID_SIZE:]:
            if name.endswith(KINDS['still']):
                self.stills.append(f'/files/stills/{name}')

    # SENDING
    async def send(self, ws: web.WebSocketResponse | None, message: dict) -> None:
        if ws is None or ws.closed:
            return
        try:
            await ws.send_str(json.dumps(message, separators=(',', ':')))
        except Exception:
            logger.debug('send failed', exc_info=True)

    async def send_bytes(self, ws: web.WebSocketResponse | None, data: bytes) -> None:
        if ws is None or ws.closed:
            return
        try:
            await ws.send_bytes(data)
        except Exception:
            logger.debug('send failed', exc_info=True)

    async def send_states(self) -> None:
        """Every phone its state: after anything that moves the line or the eye."""
        for phone, ws in list(self.phones.items()):
            await self.send(ws, self.line.state(phone))

    async def send_line(self) -> None:
        await self.send(self.smile, {'type': 'line', 'waiting': len(self.line.waiting)})

    def grid(self) -> dict:
        return {'type': 'grid', 'stills': list(reversed(self.stills))}

    @property
    def turn_ws(self) -> web.WebSocketResponse | None:
        return self.phones.get(self.line.turn) if self.line.turn else None

    # PHONES
    async def phone(self, request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse(heartbeat=HEARTBEAT, max_msg_size=MAX_FRAME + 1024)
        await ws.prepare(request)
        phone: str = uuid.uuid4().hex
        self.phones[phone] = ws
        logger.info('phone %s connected (%d online)', phone[:8], len(self.phones))
        await self.send(ws, self.grid())
        await self.send(ws, self.line.state(phone))
        try:
            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    await self.phone_message(phone, message.json())
                elif message.type == WSMsgType.BINARY:
                    data: bytes = message.data
                    # frames only from the phone with the turn, and only while SMILE can take them
                    if data and data[0] == FRAME and self.line.turn == phone and len(data) <= MAX_FRAME:
                        await self.send_bytes(self.smile, data)
                elif message.type == WSMsgType.ERROR:
                    break
        finally:
            self.phones.pop(phone, None)
            await self.phone_left(phone)
            logger.info('phone %s left (%d online)', phone[:8], len(self.phones))
        return ws

    async def phone_message(self, phone: str, message: dict) -> None:
        kind: str = str(message.get('type', ''))
        if kind == 'join':
            self.line.join(phone)
            await self.send_states()
            await self.send_line()
        elif kind == 'leave':
            await self.phone_left(phone)
        elif kind == 'timing':
            # the phone's own account of its turn's end: seconds from turn_end to the result, the play, the
            # first frame playing, the share file; what a visitor outside waits for, measured where they wait
            logger.info('phone %s timing: result %s, play %s, playing %s, file %s (%s)', phone[:8], message.get('result'),
                        message.get('play'), message.get('playing'), message.get('file'), message.get('agent', ''))
        elif kind == 'seen':
            # the phone drew the frame in flight: the newest one waiting may go; SMILE hears how long it took
            if phone == self.line.turn:
                if self._frame_in_flight is not None:
                    self._flights.append(time.monotonic() - self._frame_in_flight)
                self._frame_in_flight = None
                await self._push_frame()

    async def phone_left(self, phone: str) -> None:
        """Out of the line; a phone with the turn that leaves is `gone` to SMILE, which ends the turn."""
        if self.line.leave(phone):
            await self.send(self.smile, {'type': 'gone'})
        await self.send_states()
        await self.send_line()

    # SMILE
    async def smile_socket(self, request: web.Request) -> web.StreamResponse:
        if not self.authorized(request.query.get('token', '')):
            raise web.HTTPUnauthorized()
        if self.smile is not None and not self.smile.closed:
            logger.warning('a second SMILE connected: the first is closed')
            await self.smile.close()
        ws = web.WebSocketResponse(heartbeat=HEARTBEAT, max_msg_size=MAX_UPLOAD)
        await ws.prepare(request)
        self.smile = ws
        self.line.set_eye(True)
        logger.info('SMILE connected')
        await self.send_line()
        await self.send_states()
        try:
            async for message in ws:
                if message.type == WSMsgType.TEXT:
                    await self.smile_message(message.json())
                elif message.type == WSMsgType.BINARY:
                    data: bytes = message.data
                    if data and data[0] == SHOW:
                        self._frame_pending = data
                        await self._push_frame()
                elif message.type == WSMsgType.ERROR:
                    break
        finally:
            if self.smile is ws:
                self.smile = None
                turn = self.line.turn
                self.line.set_eye(False)
                if turn is not None:
                    await self.send(self.phones.get(turn), {'type': 'turn_end', 'reason': 'lost'})
                await self.send_states()
            logger.info('SMILE left')
        return ws

    async def _push_frame(self) -> None:
        """Send the newest screen frame to the phone with the turn when none is in flight (a phone that never
        answers, an old page, gets one every FRAME_PATIENCE seconds)."""
        if self._frame_in_flight is not None and time.monotonic() - self._frame_in_flight < FRAME_PATIENCE:
            return
        data, self._frame_pending = self._frame_pending, None
        ws = self.turn_ws
        if data is None or ws is None:
            return
        self._frame_in_flight = time.monotonic()
        await self.send_bytes(ws, data)

    def _log_flights(self) -> None:
        """At the turn's end: how the frames travelled to the phone (send to `seen`), for the log."""
        if self._flights:
            flights = sorted(self._flights)
            logger.info('frames to the phone: %d, flight median %.2f s, max %.2f s', len(flights), flights[len(flights) // 2], flights[-1])

    def _reset_frames(self) -> None:
        self._frame_pending = None
        self._frame_in_flight = None
        self._flights = []

    async def smile_message(self, message: dict) -> None:
        kind: str = str(message.get('type', ''))
        if kind == 'eye':
            self.line.set_eye(True, bool(message.get('free', False)), str(message.get('status', '')), bool(message.get('laughter', True)))
            await self.send_states()
        elif kind == 'turn_start':
            self._reset_frames()
            phone: str | None = self.line.start_turn()
            await self.send(self.smile, {'type': 'turn', 'id': phone})
            if phone is not None:
                await self.send(self.phones.get(phone), {'type': 'turn', 'window': message.get('window', 10)})
                await self.send_states()
                await self.send_line()
        elif kind in ('face', 'stage', 'music'):
            await self.send(self.turn_ws, message)
        elif kind == 'result':
            await self.result(message)
        elif kind == 'turn_end':
            self._log_flights()
            self._reset_frames()
            phone = self.line.end_turn()
            if phone is not None:
                await self.send(self.phones.get(phone), {'type': 'turn_end', 'reason': str(message.get('reason', 'done'))})
            await self.send_states()

    async def result(self, message: dict) -> None:
        """The visitor's files are uploaded (or were not): the phone of that turn gets their urls, every
        phone the new still. The turn may be over, so the phone is named; the turn's phone is the fallback."""
        id: str = self.safe_id(str(message.get('id', '')))
        result: dict = {'type': 'result', 'id': id, 'ok': bool(message.get('ok', True)),
                        'name': str(message.get('name', '')), 'caption': str(message.get('caption', ''))}
        for kind, ext in KINDS.items():
            path: str = os.path.join(self.data, kind + 's', id + ext) if id else ''
            result[kind] = f'/files/{kind}s/{id}{ext}' if path and os.path.exists(path) else None
        phone: str = str(message.get('phone', ''))
        await self.send(self.phones.get(phone) if phone else self.turn_ws, result)
        if result['still']:
            self.stills.append(result['still'])
            grid: dict = self.grid()
            for ws in list(self.phones.values()):
                await self.send(ws, grid)

    # HTTP
    async def upload(self, request: web.Request) -> web.Response:
        if not self.authorized(request.query.get('token', '')):
            raise web.HTTPUnauthorized()
        kind: str = request.query.get('kind', '')
        id: str = self.safe_id(request.query.get('id', ''))
        if kind not in UPLOAD_KINDS or not id:
            raise web.HTTPBadRequest(text='kind and id')
        ext: str = UPLOAD_KINDS[kind]
        if kind == 'laugh':
            given: str = request.query.get('ext', '').lower()
            if given not in LAUGH_EXTENSIONS:
                raise web.HTTPBadRequest(text='ext')
            ext = '.' + given
        data: bytes = await request.read()
        if not data or len(data) > MAX_UPLOAD:
            raise web.HTTPBadRequest(text='size')
        path: str = os.path.join(self.data, kind + 's', id + ext)
        temp: str = path + '.part'
        with open(temp, 'wb') as f:
            f.write(data)
        os.replace(temp, path)
        return web.json_response({'ok': True, 'url': f'/files/{kind}s/{id}{ext}'})

    async def grid_json(self, request: web.Request) -> web.Response:
        return web.json_response(self.grid())

    async def library_json(self, request: web.Request) -> web.Response:
        """The music tracks (ids) and laughter samples (file names) here: SMILE uploads what is missing,
        the phones play from it."""
        ext: str = UPLOAD_KINDS['track']
        tracks: list[str] = sorted(name[:-len(ext)] for name in os.listdir(os.path.join(self.data, 'tracks')) if name.endswith(ext))
        laughs: list[str] = sorted(name for name in os.listdir(os.path.join(self.data, 'laughs'))
                                   if name.rsplit('.', 1)[-1].lower() in LAUGH_EXTENSIONS)
        return web.json_response({'tracks': tracks, 'laughs': laughs})

    def authorized(self, token: str) -> bool:
        return bool(self.token) and secrets.compare_digest(token, self.token)

    @staticmethod
    def safe_id(id: str) -> str:
        """An id is a file name: letters, digits, `_` and `-` only."""
        return id if id and all(c.isalnum() or c in '_-' for c in id) else ''


@web.middleware
async def no_cache(request: web.Request, handler) -> web.StreamResponse:
    """The page and its files are revalidated at every load (a phone kept an old app.js for a day):
    aiohttp's static handler answers 304 from Last-Modified, so it costs nothing."""
    response: web.StreamResponse = await handler(request)
    if request.path == '/' or request.path.startswith('/web/'):
        response.headers['Cache-Control'] = 'no-cache'
    return response


def make_app(relay: Relay, web_dir: str) -> web.Application:
    app = web.Application(client_max_size=MAX_UPLOAD, middlewares=[no_cache])
    app.router.add_get('/ws/phone', relay.phone)
    app.router.add_get('/ws/smile', relay.smile_socket)
    app.router.add_post('/api/upload', relay.upload)
    app.router.add_get('/api/grid', relay.grid_json)
    app.router.add_get('/api/library', relay.library_json)
    app.router.add_static('/files/', relay.data, show_index=False)
    if os.path.isdir(web_dir):
        async def index(request: web.Request) -> web.StreamResponse:
            return web.FileResponse(os.path.join(web_dir, 'index.html'))
        app.router.add_get('/', index)
        app.router.add_static('/web/', web_dir, show_index=False)
    return app


def read_token(path: str) -> str:
    try:
        with open(path, encoding='utf-8') as f:
            return f.read().strip()
    except OSError:
        return ''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='the relay of SMILE Online')
    here: str = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8800)
    parser.add_argument('--token-file', default=os.path.join(here, 'token.txt'), help='file with the token SMILE connects with')
    parser.add_argument('--data', default=os.path.join(here, 'data'), help="folder of the visitors' files")
    parser.add_argument('--web', default=os.path.join(here, 'web'), help='folder of the phone page')
    parser.add_argument('--cert', default='', help='TLS certificate (else plain HTTP, for a proxy)')
    parser.add_argument('--key', default='')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args(argv)
    os.makedirs(args.data, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(),
                                       logging.handlers.RotatingFileHandler(os.path.join(args.data, 'relay.log'), maxBytes=5_000_000, backupCount=2, encoding='utf-8')]
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=handlers)

    token: str = read_token(args.token_file)
    if not token:
        logger.error('no token in %s: write one line into it (any long random text) and give SMILE the same', args.token_file)
        return 1
    relay = Relay(token, args.data)
    app = make_app(relay, args.web)
    context: ssl.SSLContext | None = None
    if args.cert:
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(args.cert, args.key or None)
    logger.info('relay on %s:%d (%s), files in %s', args.host, args.port, 'https' if context else 'http', args.data)
    web.run_app(app, host=args.host, port=args.port, ssl_context=context, print=None)
    return 0


if __name__ == '__main__':
    sys.exit(main())
