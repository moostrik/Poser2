"""Tests for the relay server: phones and SMILE over real websockets on a test server."""

import asyncio
import json
import os
import tempfile
import time
import unittest

from aiohttp import ClientWebSocketResponse, web
from aiohttp.test_utils import TestClient, TestServer

from apps.smile.relay.server import FRAME, KEEP_AFTER_LEAVE, KEEP_AT_MOST, LAST, SHOW, Relay, make_app

TOKEN: str = 'test-token'
AUTH: dict[str, str] = {'Authorization': f'Bearer {TOKEN}'}
RANDOM_NAME: str = r'^/files/{}s/v1-[A-Za-z0-9_-]{{16}}\.{}$'       # a visitor's file: the id, a random part, the extension


class RelayTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.relay = Relay(TOKEN, self.folder.name)
        self.client = await self.start_client(os.path.join(self.folder.name, 'no-web'))
        # the phone scanned the code: its pass is the cookie / set from the url, on every request from here
        self.client.session.cookie_jar.update_cookies({'pass': self.relay.current_pass})

    async def start_client(self, web_dir: str) -> TestClient:
        client = TestClient(TestServer(make_app(self.relay, web_dir)))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        return client

    async def bare_client(self) -> TestClient:
        """A client without the pass: the same server, its own cookies."""
        client = TestClient(self.client.server)
        await client.start_server()
        self.addAsyncCleanup(client.close)
        return client

    async def phone(self, client: TestClient | None = None) -> ClientWebSocketResponse:
        ws = await (client or self.client).ws_connect('/ws/phone')
        self.assertEqual((await ws.receive_json())['type'], 'state')
        return ws

    async def smile(self) -> ClientWebSocketResponse:
        ws = await self.client.ws_connect('/ws/smile', headers=AUTH)
        self.assertEqual((await ws.receive_json())['type'], 'pass')
        self.assertEqual((await ws.receive_json())['type'], 'line')
        return ws

    async def receive(self, ws: ClientWebSocketResponse, type: str, **match) -> dict:
        """The next message of that type with those values (the others in between are skipped)."""
        for _ in range(20):
            message = await asyncio.wait_for(ws.receive_json(), 2.0)
            if message['type'] == type and all(message.get(k) == v for k, v in match.items()):
                return message
        self.fail(f'no {type} message')

    async def receive_bytes(self, ws: ClientWebSocketResponse) -> bytes:
        """The next binary message (the texts in between are skipped)."""
        for _ in range(20):
            message = await asyncio.wait_for(ws.receive(), 2.0)
            if message.type == web.WSMsgType.BINARY:
                return message.data
        self.fail('no binary message')

    async def upload(self, kind: str, id: str = 'v1', data: bytes = b'x') -> str:
        """SMILE's upload of a file; its url."""
        response = await self.client.post(f'/api/upload?kind={kind}&id={id}', headers=AUTH, data=data)
        self.assertEqual(response.status, 200)
        return (await response.json())['url']

    async def exists(self, url: str) -> bool:
        async with self.client.get(url) as response:
            await response.read()       # the file is closed again (Windows refuses to delete an open one)
            return response.status == 200

    async def test_the_page_is_not_cached(self):
        web_dir = os.path.join(self.folder.name, 'web')
        os.mkdir(web_dir)
        for name in ('index.html', 'app.js', 'expired.html'):
            with open(os.path.join(web_dir, name), 'w') as f:
                f.write(name)
        client = await self.start_client(web_dir)
        client.session.cookie_jar.update_cookies({'pass': self.relay.current_pass})
        for path in ('/', '/web/app.js'):
            response = await client.get(path)
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers.get('Cache-Control'), 'no-cache')
        response = await client.get('/api/library')
        self.assertIsNone(response.headers.get('Cache-Control'))

    async def test_the_page_opens_with_the_codes_pass_and_keeps_it_as_a_cookie(self):
        web_dir = os.path.join(self.folder.name, 'web')
        os.mkdir(web_dir)
        for name in ('index.html', 'expired.html'):
            with open(os.path.join(web_dir, name), 'w') as f:
                f.write(name)
        client = await self.start_client(web_dir)       # a phone that scanned nothing yet
        response = await client.get('/')
        self.assertEqual((response.status, await response.text()), (403, 'expired.html'))
        response = await client.get('/?pass=wrong')
        self.assertEqual(response.status, 403)
        self.assertEqual((await client.get('/api/library')).status, 403)
        # the code's url: the page, and the pass as a cookie for everything after it
        response = await client.get(f'/?pass={self.relay.current_pass}')
        self.assertEqual((response.status, await response.text()), (200, 'index.html'))
        self.assertIn('pass=', response.headers.get('Set-Cookie', ''))
        self.assertEqual((await client.get('/api/library')).status, 200)
        track = await self.upload('track', id='0001')
        self.assertEqual((await client.get(track)).status, 200)
        self.assertEqual((await (await self.bare_client()).get(track)).status, 403)
        phone = await self.phone(client)                # the websocket too
        await phone.close()

    async def test_a_phone_whose_pass_ran_out_is_told_so_and_closed(self):
        bare = await self.bare_client()
        ws = await bare.ws_connect('/ws/phone')
        self.assertEqual((await ws.receive_json())['type'], 'expired')
        self.assertEqual((await ws.receive()).type, web.WSMsgType.CLOSE)
        # a tab kept open: its pass runs out meanwhile, the join is refused
        phone = await self.phone()
        self.relay.rotate(time.time() + self.relay.pass_lifetime + 1)
        await phone.send_json({'type': 'join'})
        self.assertEqual((await phone.receive_json())['type'], 'expired')
        self.assertEqual((await phone.receive()).type, web.WSMsgType.CLOSE)
        self.assertEqual(self.relay.line.waiting, [])

    async def test_passes_rotate_and_survive_a_restart(self):
        first = self.relay.current_pass
        second = self.relay.rotate()
        self.assertTrue(self.relay.valid_pass(first) and self.relay.valid_pass(second))
        self.assertNotEqual(first, second)
        later = time.time() + self.relay.pass_lifetime + 1
        self.assertFalse(self.relay.valid_pass(first, later))
        third = self.relay.rotate(later)             # the old ones go at a rotation
        self.assertEqual([v for v, _ in self.relay.passes], [third])
        # a new relay on the same data folder knows the passes it issued
        again = Relay(TOKEN, self.folder.name)
        self.assertEqual(again.current_pass, third)
        self.assertTrue(again.valid_pass(third, later))
        # SMILE hears the newest at connect
        smile = await self.smile()
        await smile.close()

    async def test_smile_needs_the_token_in_the_header(self):
        response = await self.client.get('/ws/smile', headers={'Upgrade': 'websocket', 'Authorization': 'Bearer wrong'})
        self.assertEqual(response.status, 401)
        response = await self.client.post('/api/upload?kind=portrait&id=x', data=b'x')
        self.assertEqual(response.status, 401)
        # the token in the URL (the old way: a request line in a log would keep it) is no token at all
        response = await self.client.post(f'/api/upload?token={TOKEN}&kind=portrait&id=x', data=b'x')
        self.assertEqual(response.status, 401)

    async def test_a_phone_joins_and_smile_hears_the_line(self):
        phone = await self.phone()
        smile = await self.smile()
        await smile.send_json({'type': 'eye', 'free': True, 'status': 'looking for faces', 'laughter': False})
        self.assertFalse((await self.receive(phone, 'state', eye='free'))['laughter'])     # the room's switch, for the phones too
        await phone.send_json({'type': 'join'})
        await self.receive(phone, 'state', eye='waiting')
        self.assertEqual((await self.receive(smile, 'line'))['waiting'], 1)

    async def test_a_turn_passes_frames_face_data_and_the_result(self):
        phone = await self.phone()
        smile = await self.smile()
        await phone.send_json({'type': 'join'})
        await self.receive(smile, 'line')
        await smile.send_json({'type': 'turn_start', 'window': 10})
        turn = await self.receive(smile, 'turn')
        self.assertIsNotNone(turn['id'])
        self.assertEqual((await self.receive(phone, 'turn'))['window'], 10)
        # frames go to SMILE, face data and images come back
        await phone.send_bytes(bytes([FRAME]) + b'jpeg')
        self.assertEqual(await self.receive_bytes(smile), bytes([FRAME]) + b'jpeg')
        await smile.send_json({'type': 'face', 'charge': 0.5})
        self.assertEqual((await self.receive(phone, 'face'))['charge'], 0.5)
        await smile.send_json({'type': 'music', 'id': '0001', 'url': '/files/tracks/0001.m4a', 'offset': 0})
        self.assertEqual((await self.receive(phone, 'music'))['url'], '/files/tracks/0001.m4a')
        await smile.send_bytes(bytes([SHOW]) + b'img')
        self.assertEqual(await self.receive_bytes(phone), bytes([SHOW]) + b'img')
        # the next frames wait until the phone has seen that one, and only the newest of them goes
        await smile.send_bytes(bytes([SHOW]) + b'img2')
        await smile.send_bytes(bytes([SHOW]) + b'img3')
        await smile.send_json({'type': 'stage', 'stage': 'grin', 'progress': 0.1})
        self.assertEqual((await self.receive(phone, 'stage'))['stage'], 'grin')      # not held behind the frames
        await phone.send_json({'type': 'seen'})
        self.assertEqual(await self.receive_bytes(phone), bytes([SHOW]) + b'img3')
        # the turn's last frame goes the same way
        await smile.send_bytes(bytes([LAST]) + b'end')
        await phone.send_json({'type': 'seen'})
        self.assertEqual(await self.receive_bytes(phone), bytes([LAST]) + b'end')
        # the turn ends with the sequence; the files and the result follow, addressed to the phone
        await smile.send_json({'type': 'turn_end', 'reason': 'done'})
        self.assertEqual((await self.receive(phone, 'turn_end'))['reason'], 'done')
        self.assertIsNone(self.relay.line.turn)
        portrait = await self.upload('portrait', data=b'portrait')
        self.assertRegex(portrait, RANDOM_NAME.format('portrait', 'jpg'))
        await smile.send_json({'type': 'result', 'id': 'v1', 'phone': turn['id'], 'name': 'Name', 'caption': 'hi', 'ok': True})
        result = await self.receive(phone, 'result')
        self.assertEqual((result['name'], result['caption'], result['ok']), ('Name', 'hi', True))
        self.assertEqual(result['portrait'], portrait)
        self.assertIsNone(result['grin'])
        response = await self.client.get(portrait)
        self.assertEqual(await response.read(), b'portrait')
        # a result without files (the visitor was not kept) still reaches the phone, so it stops waiting
        await smile.send_json({'type': 'result', 'id': '', 'phone': turn['id'], 'name': 'Name', 'caption': 'hi', 'ok': False})
        result = await self.receive(phone, 'result')
        self.assertFalse(result['ok'])
        self.assertIsNone(result['portrait'])

    async def test_frames_from_a_phone_without_the_turn_are_dropped(self):
        phone = await self.phone()
        smile = await self.smile()
        await phone.send_bytes(bytes([FRAME]) + b'jpeg')
        await phone.send_json({'type': 'join'})
        message = await asyncio.wait_for(smile.receive(), 2.0)     # the next thing SMILE gets is the line, not the frame
        self.assertEqual(message.type, web.WSMsgType.TEXT)
        self.assertEqual(json.loads(message.data)['type'], 'line')

    async def test_a_phone_that_leaves_during_its_turn_is_gone(self):
        phone = await self.phone()
        smile = await self.smile()
        await phone.send_json({'type': 'join'})
        await smile.send_json({'type': 'turn_start'})
        await self.receive(smile, 'turn')
        await phone.close()
        self.assertEqual((await self.receive(smile, 'gone'))['type'], 'gone')
        self.assertIsNone(self.relay.line.turn)

    async def test_smile_leaving_puts_everyone_to_sleep(self):
        phone = await self.phone()
        smile = await self.smile()
        await phone.send_json({'type': 'join'})
        await smile.send_json({'type': 'turn_start'})
        await self.receive(smile, 'turn')
        await smile.close()
        self.assertEqual((await self.receive(phone, 'turn_end'))['reason'], 'lost')
        await self.receive(phone, 'state', eye='asleep')

    async def test_the_library_is_uploaded_once_and_listed(self):
        response = await self.client.get('/api/library')
        self.assertEqual(await response.json(), {'tracks': [], 'laughs': []})
        response = await self.client.post('/api/upload?kind=track&id=0001', headers=AUTH, data=b'aac')
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())['url'], '/files/tracks/0001.m4a')
        # a laugh keeps its own extension, mp3 or wav
        response = await self.client.post('/api/upload?kind=laugh&id=Hehe-2&ext=wav', headers=AUTH, data=b'wav')
        self.assertEqual((await response.json())['url'], '/files/laughs/Hehe-2.wav')
        response = await self.client.post('/api/upload?kind=laugh&id=Haha', headers=AUTH, data=b'x')
        self.assertEqual(response.status, 400)
        response = await self.client.post('/api/upload?kind=laugh&id=Haha&ext=exe', headers=AUTH, data=b'x')
        self.assertEqual(response.status, 400)
        response = await self.client.get('/api/library')
        self.assertEqual(await response.json(), {'tracks': ['0001'], 'laughs': ['Hehe-2.wav']})
        response = await self.client.get('/files/tracks/0001.m4a')
        self.assertEqual(await response.read(), b'aac')

    async def test_a_visitors_files_go_minutes_after_their_phone_left(self):
        phone = await self.phone()
        smile = await self.smile()
        await phone.send_json({'type': 'join'})
        await smile.send_json({'type': 'turn_start'})
        turn = await self.receive(smile, 'turn')
        portrait = await self.upload('portrait')
        grin = await self.upload('grin')
        self.assertRegex(grin, RANDOM_NAME.format('grin', 'mp4'))
        await smile.send_json({'type': 'result', 'id': 'v1', 'phone': turn['id'], 'ok': True})
        result = await self.receive(phone, 'result')
        self.assertEqual((result['portrait'], result['grin']), (portrait, grin))
        # the phone stays: its files stay, however often the sweep runs
        now = time.time()
        self.relay.sweep(now + KEEP_AFTER_LEAVE + 1)
        self.assertTrue(await self.exists(grin))
        # the phone leaves (the tab closed): the files have their last minutes, then go
        await phone.close()
        await self.receive(smile, 'gone')
        self.relay.sweep(now + KEEP_AFTER_LEAVE - 10)
        self.assertTrue(await self.exists(grin))
        self.relay.sweep(now + KEEP_AFTER_LEAVE + 1)
        self.assertFalse(await self.exists(grin))
        self.assertFalse(await self.exists(portrait))
        self.assertEqual(self.relay.deliveries, {})

    async def test_a_result_for_a_phone_that_is_gone_starts_the_last_minutes_at_once(self):
        smile = await self.smile()
        grin = await self.upload('grin')
        await smile.send_json({'type': 'result', 'id': 'v1', 'phone': 'gone', 'ok': True})
        await smile.send_json({'type': 'turn_start'})
        await self.receive(smile, 'turn')       # answered after the result: the result has been handled
        now = time.time()
        self.relay.sweep(now + KEEP_AFTER_LEAVE + 1)
        self.assertFalse(await self.exists(grin))

    async def test_the_sweep_takes_old_files_and_leaves_the_library(self):
        smile = await self.smile()
        grin = await self.upload('grin')        # a delivery whose phone never leaves: KEEP_AT_MOST ends it
        track = await self.upload('track', id='0001')
        old = os.path.join(self.folder.name, 'portraits', 'older-relay.jpg')      # a file no delivery knows
        with open(old, 'wb') as f:
            f.write(b'x')
        os.utime(old, (time.time() - KEEP_AT_MOST - 1, time.time() - KEEP_AT_MOST - 1))
        self.relay.sweep()
        self.assertFalse(os.path.exists(old))
        self.assertTrue(await self.exists(grin))
        self.relay.sweep(time.time() + KEEP_AT_MOST + 1)
        self.assertFalse(await self.exists(grin))
        self.assertTrue(await self.exists(track))

    async def test_an_upload_needs_a_safe_id_and_kind(self):
        response = await self.client.post('/api/upload?kind=portrait&id=../x', headers=AUTH, data=b'x')
        self.assertEqual(response.status, 400)
        response = await self.client.post('/api/upload?kind=exe&id=x', headers=AUTH, data=b'x')
        self.assertEqual(response.status, 400)


if __name__ == '__main__':
    unittest.main()
