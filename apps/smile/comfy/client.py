"""Minimal ComfyUI API client: HTTP for upload/queue/view, a websocket for events and previews."""

import json
import struct
import uuid
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import requests
import websocket

PREVIEW_IMAGE: int = 1
PREVIEW_IMAGE_WITH_METADATA: int = 4


@dataclass
class Event:
    type: str                                   # websocket message type, or 'preview'
    data: dict = field(default_factory=dict)
    image: np.ndarray | None = None             # BGR, only for previews


def decode(data: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


class ComfyClient():
    def __init__(self, host: str, port: int) -> None:
        self.base: str = f'http://{host}:{port}'
        self.ws_url: str = f'ws://{host}:{port}/ws'
        self.client_id: str = str(uuid.uuid4())
        self.ws: websocket.WebSocket | None = None

    # CONNECTION
    def connect(self) -> None:
        """Raises when the server is not reachable (yet)."""
        requests.get(f'{self.base}/system_stats', timeout=2).raise_for_status()
        ws = websocket.WebSocket()
        ws.connect(f'{self.ws_url}?clientId={self.client_id}', timeout=5)
        # previews then carry the id of the node that made them
        ws.send(json.dumps({'type': 'feature_flags', 'data': {'supports_preview_metadata': True}}))
        self.ws = ws

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
        self.ws = None

    @property
    def connected(self) -> bool:
        return self.ws is not None and self.ws.connected

    # HTTP
    def upload(self, image: np.ndarray, name: str) -> str:
        ok, png = cv2.imencode('.png', image)
        if not ok:
            raise ValueError('could not encode image')
        response = requests.post(f'{self.base}/upload/image', files={'image': (name, png.tobytes(), 'image/png')},
                                 data={'overwrite': 'true'}, timeout=10)
        response.raise_for_status()
        result: dict = response.json()
        return f"{result['subfolder']}/{result['name']}" if result.get('subfolder') else result['name']

    def queue(self, workflow: dict) -> str:
        response = requests.post(f'{self.base}/prompt', json={'prompt': workflow, 'client_id': self.client_id}, timeout=10)
        result: dict = response.json()
        # ComfyUI skips outputs with invalid inputs and still runs the rest; treat that as an error
        if response.status_code != 200 or 'prompt_id' not in result or result.get('node_errors'):
            raise RuntimeError(f'ComfyUI refused the workflow: {json.dumps(result)[:500]}')
        return result['prompt_id']

    def view(self, image_info: dict) -> np.ndarray | None:
        response = requests.get(f'{self.base}/view', params={'filename': image_info['filename'],
                                                              'subfolder': image_info.get('subfolder', ''),
                                                              'type': image_info.get('type', 'output')}, timeout=10)
        response.raise_for_status()
        return decode(response.content)

    def object_info(self, class_type: str) -> dict:
        """The input definitions of one node type, as this server has it."""
        response = requests.get(f'{self.base}/object_info/{class_type}', timeout=10)
        response.raise_for_status()
        return response.json()[class_type]

    def busy(self) -> bool:
        """True while the server has a job running or waiting."""
        response = requests.get(f'{self.base}/queue', timeout=5)
        response.raise_for_status()
        queue: dict = response.json()
        return bool(queue.get('queue_running') or queue.get('queue_pending'))

    def interrupt(self) -> None:
        requests.post(f'{self.base}/queue', json={'clear': True}, timeout=5)
        requests.post(f'{self.base}/interrupt', timeout=5)

    # WEBSOCKET
    def receive(self, timeout: float) -> Event | None:
        """Next event, or None when nothing arrived within *timeout* seconds."""
        if self.ws is None:
            return None
        self.ws.settimeout(timeout)
        try:
            message: Any = self.ws.recv()
        except websocket.WebSocketTimeoutException:
            return None
        if isinstance(message, bytes):
            return self._binary(message)
        parsed: dict = json.loads(message)
        return Event(parsed.get('type', ''), parsed.get('data', {}) or {})

    def _binary(self, message: bytes) -> Event | None:
        event_type: int = struct.unpack('>I', message[:4])[0]
        if event_type == PREVIEW_IMAGE_WITH_METADATA:
            length: int = struct.unpack('>I', message[4:8])[0]
            metadata: dict = json.loads(message[8:8 + length])
            return Event('preview', metadata, decode(message[8 + length:]))
        if event_type == PREVIEW_IMAGE:
            return Event('preview', {}, decode(message[8:]))
        return None
