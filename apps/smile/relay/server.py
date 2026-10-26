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
    show images.
SMILE -> relay: eye {free, status}, turn_start, face, stage, result {id, name, caption}, turn_end {reason};
    show images; uploads over HTTP.
Relay -> SMILE: line {waiting}, turn {id}, gone (the phone with the turn left); frames.

Phones only open their camera on HTTPS: run it behind `tailscale serve`, a reverse proxy, or with
`--cert` and `--key`."""

import argparse
import asyncio
import json
import logging
import os
import secrets
import ssl
import sys
import uuid
from collections import deque
from typing import Any

from aiohttp import WSMsgType, web

from relay.line import Line

logger = logging.getLogger('relay')

FRAME: int = 0      # a phone's camera frame, to SMILE
SHOW: int = 1       # an image from SMILE, to the phone with the turn

KINDS: dict[str, str] = {'still': '.jpg', 'portrait': '.jpg', 'grin': '.mp4'}     # uploads, and their folders (kind + 's')
MAX_FRAME: int = 512 * 1024             # bytes; a phone frame larger than this is dropped
MAX_UPLOAD: int = 64 * 1024 * 1024
GRID_SIZE: int = 32                     # stills the phones are told about
HEARTBEAT: float = 20.0                 # s; a websocket ping, so a dead phone leaves the line


class Relay:
    def __init__(self, token: str, data: str) -> None:
        self.token: str = token
        self.data: str = data
        self.line: Line = Line()
        self.phones: dict[str, web.WebSocketResponse] = {}
        self.smile: web.WebSocketResponse | None = None
        self.stills: deque[str] = deque(maxlen=GRID_SIZE)      # urls, the newest last
        for kind, ext in KINDS.items():
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
                        await self.send_bytes(self.turn_ws, data)
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

    async def smile_message(self, message: dict) -> None:
        kind: str = str(message.get('type', ''))
        if kind == 'eye':
            self.line.set_eye(True, bool(message.get('free', False)), str(message.get('status', '')))
            await self.send_states()
        elif kind == 'turn_start':
            phone: str | None = self.line.start_turn()
            await self.send(self.smile, {'type': 'turn', 'id': phone})
            if phone is not None:
                await self.send(self.phones.get(phone), {'type': 'turn', 'window': message.get('window', 10)})
                await self.send_states()
                await self.send_line()
        elif kind in ('face', 'stage'):
            await self.send(self.turn_ws, message)
        elif kind == 'result':
            await self.result(message)
        elif kind == 'turn_end':
            phone = self.line.end_turn()
            if phone is not None:
                await self.send(self.phones.get(phone), {'type': 'turn_end', 'reason': str(message.get('reason', 'done'))})
            await self.send_states()

    async def result(self, message: dict) -> None:
        """The visitor's files are uploaded: the phone gets their urls, every phone the new still."""
        id: str = self.safe_id(str(message.get('id', '')))
        if not id:
            return
        result: dict = {'type': 'result', 'id': id, 'name': str(message.get('name', '')), 'caption': str(message.get('caption', ''))}
        for kind, ext in KINDS.items():
            path: str = os.path.join(self.data, kind + 's', id + ext)
            result[kind] = f'/files/{kind}s/{id}{ext}' if os.path.exists(path) else None
        await self.send(self.turn_ws, result)
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
        if kind not in KINDS or not id:
            raise web.HTTPBadRequest(text='kind and id')
        data: bytes = await request.read()
        if not data or len(data) > MAX_UPLOAD:
            raise web.HTTPBadRequest(text='size')
        path: str = os.path.join(self.data, kind + 's', id + KINDS[kind])
        temp: str = path + '.part'
        with open(temp, 'wb') as f:
            f.write(data)
        os.replace(temp, path)
        return web.json_response({'ok': True, 'url': f'/files/{kind}s/{id}{KINDS[kind]}'})

    async def grid_json(self, request: web.Request) -> web.Response:
        return web.json_response(self.grid())

    def authorized(self, token: str) -> bool:
        return bool(self.token) and secrets.compare_digest(token, self.token)

    @staticmethod
    def safe_id(id: str) -> str:
        """An id is a file name: letters, digits, `_` and `-` only."""
        return id if id and all(c.isalnum() or c in '_-' for c in id) else ''


def make_app(relay: Relay, web_dir: str) -> web.Application:
    app = web.Application(client_max_size=MAX_UPLOAD)
    app.router.add_get('/ws/phone', relay.phone)
    app.router.add_get('/ws/smile', relay.smile_socket)
    app.router.add_post('/api/upload', relay.upload)
    app.router.add_get('/api/grid', relay.grid_json)
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
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

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
