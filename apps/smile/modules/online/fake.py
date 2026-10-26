"""A stand-in for the installation, to test the relay and the phone page without SMILE:

    python -m modules.online.fake [--relay ws://127.0.0.1:8800/ws/smile] [--token-file data/online_token.txt]

It connects like SMILE, says its eye is free, takes the first in line, judges their frames with
SMILE's own face detector (SCRFD; without its model, the middle of the frame counts as a face: a box,
its distance from good by its size and place, the charge rising while it is good), flashes, sends the
frame back through a few fake stages and uploads it as the portrait and the still. No grin: the page
then shows the portrait."""

import argparse
import logging
import os
import tempfile
import time
from threading import Lock, Thread

import cv2
import numpy as np

from modules.online.Link import Link, read_token
from modules.online.Online import encode

logger = logging.getLogger('fake-smile')

MODEL: str = 'data/models/scrfd_10g_bnkps.onnx'

CHARGE_SECONDS: float = 3.0
WINDOW: float = 15.0
STAGES: tuple[str, ...] = ('depth', 'describe', 'diffusion', 'portrait', 'grin')


class FakeSmile:
    def __init__(self, link_url: str, token: str) -> None:
        self.link: Link = Link(link_url, token, self.on_message, self.on_frame, name='fake-smile')
        self.detector = None
        try:
            from modules.face.FaceDetector import FaceDetector
            self.detector = FaceDetector(MODEL)
        except Exception as e:
            logger.warning('no face detector (%s): the middle of every frame counts as a face', e)
        self.mutex: Lock = Lock()
        self.waiting: int = 0
        self.turn: bool = False
        self.busy: bool = False     # the transformation runs
        self.turn_start: float = 0.0
        self.charge: float = 0.0
        self.charged_at: float = 0.0
        self.last_frame: np.ndarray | None = None
        self.asked: float = 0.0

    def on_message(self, message: dict) -> None:
        kind: str = message.get('type', '')
        with self.mutex:
            if kind == 'line':
                self.waiting = int(message.get('waiting', 0))
            elif kind == 'turn':
                if message.get('id'):
                    self.turn = True
                    self.turn_start = time.time()
                    self.charge = 0.0
                    self.charged_at = 0.0
                    logger.info('turn starts')
                self.asked = 0.0
            elif kind == 'gone':
                if self.turn and not self.busy:
                    self.turn = False
                    self.link.send({'type': 'turn_end', 'reason': 'lost'})

    def on_frame(self, jpeg: bytes) -> None:
        with self.mutex:
            if not self.turn or self.busy:
                return
        image: np.ndarray | None = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        self.last_frame = image
        h, w = image.shape[:2]
        found: list[tuple[float, float, float, float]]     # x, y, width, height of each face
        if self.detector is not None:
            found = [(float(f.bbox[0]), float(f.bbox[1]), f.width, f.height) for f in self.detector.detect(image)]
        else:
            found = [(w * 0.35, h * 0.3, w * 0.3, h * 0.3)]
        faces: list[dict] = []
        now: float = time.time()
        good: bool = False
        for i, (x, y, fw, fh) in enumerate(sorted(found, key=lambda f: -f[2] * f[3])):
            # the portrait crop around the face, like the installation's, and how far from good:
            # too small (far away) or too near an edge
            cx, cy = x + fw / 2, y + fh / 2 - fh * 0.1
            cw, ch = fw * 2.2, fw * 2.2 * 1.25
            box = [max(0.0, (cx - cw / 2) / w), max(0.0, (cy - ch / 2) / h), min(1.0, (cx + cw / 2) / w), min(1.0, (cy + ch / 2) / h)]
            size: float = fw / w
            edge: float = min(cx - cw / 2, cy - ch / 2, w - cx - cw / 2, h - cy - ch / 2) / cw
            reject: str | None = None
            distance: float = 0.0
            if size < 0.18:
                reject, distance = f'eyes {size:.2f} < 0.18', min(1.0, (0.18 - size) / 0.1)
            elif edge < -0.1:
                reject, distance = f'edge {edge:.2f} < -0.10', min(1.0, (-0.1 - edge) / 0.2)
            subject: bool = i == 0
            if subject and reject is None:
                good = True
            faces.append({'box': [round(v, 4) for v in box], 'distance': round(distance, 3), 'subject': subject, 'reject': reject})
        with self.mutex:
            if good:
                if self.charged_at == 0.0:
                    self.charged_at = now
                self.charge = min(1.0, (now - self.charged_at) / CHARGE_SECONDS)
            else:
                self.charged_at = 0.0
                self.charge = max(0.0, self.charge - 0.05)
            charge: float = self.charge
        self.link.send({'type': 'face', 'charge': round(charge, 3), 'faces': faces})
        if charge >= 1.0:
            # on its own thread: this one is the link's, which has to keep sending
            Thread(target=self.transform, args=(image,), daemon=True).start()

    def transform(self, image: np.ndarray) -> None:
        """The flash and a fake transformation: the frame itself, worked over a little per stage."""
        with self.mutex:
            if not self.turn or self.busy:
                return
            self.busy = True        # the turn goes on until the result is delivered: no judging, no new turn
        link: Link = self.link
        link.send({'type': 'stage', 'stage': 'flash'})
        time.sleep(1.0)
        link.send({'type': 'stage', 'stage': 'depth'})
        depth: np.ndarray = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        link.send_image(encode(cv2.GaussianBlur(depth, (31, 31), 0)) or b'')
        time.sleep(1.5)
        link.send({'type': 'stage', 'stage': 'describe', 'text': 'a person, looking at a phone, hopeful'})
        for step in range(1, 7):
            time.sleep(0.5)
            blur: int = 2 * (7 - step) * 4 + 1
            link.send({'type': 'stage', 'stage': 'diffusion', 'progress': round(step / 6, 2)})
            link.send_image(encode(cv2.GaussianBlur(image, (blur, blur), 0)) or b'')
        portrait: np.ndarray = cv2.applyColorMap(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_PINK)
        link.send({'type': 'stage', 'stage': 'portrait'})
        link.send_image(encode(portrait) or b'')
        time.sleep(1.0)
        link.send({'type': 'stage', 'stage': 'grin'})
        id: str = time.strftime('%Y_%m_%d_%H_%M_%S')
        path: str = os.path.join(tempfile.gettempdir(), f'fake_smile_{id}.jpg')
        cv2.imwrite(path, portrait)

        def done(ok: bool) -> None:
            if ok:
                link.send({'type': 'result', 'id': id, 'name': 'Matthias Oostrik', 'caption': 'loves your smile!'})
                link.send({'type': 'turn_end', 'reason': 'done'})
            else:
                link.send({'type': 'turn_end', 'reason': 'reset'})
            with self.mutex:
                self.turn = False
                self.busy = False
            os.remove(path)
            logger.info('turn ends (%s)', 'done' if ok else 'upload failed')
        link.upload([('portrait', id, path), ('still', id, path)], done)

    def run(self) -> None:
        self.link.start()
        while True:
            time.sleep(0.5)
            if not self.link.connected:
                continue
            now: float = time.time()
            with self.mutex:
                turn: bool = self.turn
                waiting: int = self.waiting
                asked: float = self.asked
                if turn and not self.busy and now - self.turn_start > WINDOW and self.charge == 0.0:
                    self.turn = False
                    self.link.send({'type': 'turn_end', 'reason': 'missed'})
                    logger.info('turn missed')
                    turn = False
            self.link.send({'type': 'eye', 'free': not turn, 'status': 'looking for faces'})
            if not turn and waiting > 0 and now - asked > 3.0:
                with self.mutex:
                    self.asked = now
                self.link.send({'type': 'turn_start', 'window': WINDOW})


def main() -> None:
    parser = argparse.ArgumentParser(description='a stand-in for SMILE at the relay')
    parser.add_argument('--relay', default='ws://127.0.0.1:8800/ws/smile')
    parser.add_argument('--token-file', default='data/online_token.txt')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    token: str = read_token(args.token_file)
    if not token:
        parser.error(f'no token in {args.token_file}')
    FakeSmile(args.relay, token).run()


if __name__ == '__main__':
    main()
