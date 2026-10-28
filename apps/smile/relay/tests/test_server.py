"""Tests for the relay server: phones and SMILE over real websockets on a test server."""

import asyncio
import json
import os
import tempfile
import unittest

from aiohttp import ClientWebSocketResponse, web
from aiohttp.test_utils import TestClient, TestServer

from apps.smile.relay.server import FRAME, LAST, SHOW, Relay, make_app

TOKEN: str = 'test-token'


class RelayTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.relay = Relay(TOKEN, self.folder.name)
        self.client = TestClient(TestServer(make_app(self.relay, os.path.join(self.folder.name, 'no-web'))))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def phone(self) -> ClientWebSocketResponse:
        ws = await self.client.ws_connect('/ws/phone')
        self.assertEqual((await ws.receive_json())['type'], 'grid')
        self.assertEqual((await ws.receive_json())['type'], 'state')
        return ws

    async def smile(self) -> ClientWebSocketResponse:
        ws = await self.client.ws_connect(f'/ws/smile?token={TOKEN}')
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

    async def test_the_page_is_not_cached(self):
        web_dir = os.path.join(self.folder.name, 'web')
        os.mkdir(web_dir)
        for name in ('index.html', 'app.js'):
            with open(os.path.join(web_dir, name), 'w') as f:
                f.write(name)
        client = TestClient(TestServer(make_app(self.relay, web_dir)))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        for path in ('/', '/web/app.js'):
            response = await client.get(path)
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers.get('Cache-Control'), 'no-cache')
        response = await client.get('/api/grid')
        self.assertIsNone(response.headers.get('Cache-Control'))

    async def test_smile_needs_the_token(self):
        response = await self.client.get('/ws/smile?token=wrong', headers={'Upgrade': 'websocket'})
        self.assertEqual(response.status, 401)
        response = await self.client.post('/api/upload?token=&kind=still&id=x', data=b'x')
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
        other = await self.phone()
        response = await self.client.post(f'/api/upload?token={TOKEN}&kind=still&id=v1', data=b'still')
        self.assertEqual(response.status, 200)
        await smile.send_json({'type': 'result', 'id': 'v1', 'phone': turn['id'], 'name': 'Name', 'caption': 'hi', 'ok': True})
        result = await self.receive(phone, 'result')
        self.assertEqual((result['name'], result['caption'], result['ok']), ('Name', 'hi', True))
        self.assertEqual(result['still'], '/files/stills/v1.jpg')
        self.assertIsNone(result['grin'])
        self.assertEqual((await self.receive(other, 'grid'))['stills'], ['/files/stills/v1.jpg'])
        response = await self.client.get('/files/stills/v1.jpg')
        self.assertEqual(await response.read(), b'still')
        # a result without files (the visitor was not kept) still reaches the phone, so it stops waiting
        await smile.send_json({'type': 'result', 'id': '', 'phone': turn['id'], 'name': 'Name', 'caption': 'hi', 'ok': False})
        result = await self.receive(phone, 'result')
        self.assertFalse(result['ok'])
        self.assertIsNone(result['still'])

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
        response = await self.client.post(f'/api/upload?token={TOKEN}&kind=track&id=0001', data=b'aac')
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())['url'], '/files/tracks/0001.m4a')
        # a laugh keeps its own extension, mp3 or wav
        response = await self.client.post(f'/api/upload?token={TOKEN}&kind=laugh&id=Hehe-2&ext=wav', data=b'wav')
        self.assertEqual((await response.json())['url'], '/files/laughs/Hehe-2.wav')
        response = await self.client.post(f'/api/upload?token={TOKEN}&kind=laugh&id=Haha', data=b'x')
        self.assertEqual(response.status, 400)
        response = await self.client.post(f'/api/upload?token={TOKEN}&kind=laugh&id=Haha&ext=exe', data=b'x')
        self.assertEqual(response.status, 400)
        response = await self.client.get('/api/library')
        self.assertEqual(await response.json(), {'tracks': ['0001'], 'laughs': ['Hehe-2.wav']})
        response = await self.client.get('/files/tracks/0001.m4a')
        self.assertEqual(await response.read(), b'aac')
        # a track is not one of a visitor's files
        self.assertNotIn('track', (await self.client.get('/api/grid')).headers)

    async def test_an_upload_needs_a_safe_id_and_kind(self):
        response = await self.client.post(f'/api/upload?token={TOKEN}&kind=still&id=../x', data=b'x')
        self.assertEqual(response.status, 400)
        response = await self.client.post(f'/api/upload?token={TOKEN}&kind=exe&id=x', data=b'x')
        self.assertEqual(response.status, 400)


if __name__ == '__main__':
    unittest.main()
