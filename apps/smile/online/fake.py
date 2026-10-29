"""A stand-in for the installation, to test the relay and the phone page without SMILE:

    python -m apps.smile.online.fake [--relay ws://127.0.0.1:8800/ws/smile] [--token-file data/online_token.txt]

It connects like SMILE, says its eye is free, takes the first in line, judges their frames with
SMILE's own face detector (SCRFD; without its model, the middle of the frame counts as a face: a box,
its distance from good by its size and place, the charge rising while it is good), flashes, streams a
fake transformation of the frame as the screen (a few seconds at 10 fps), ends the turn and then
uploads the frame as the portrait, the result addressed to the phone. No grin: the page then shows
the portrait."""

import argparse
import logging
import os
import tempfile
import time
from threading import Lock, Thread

import cv2
import numpy as np

from .link import Link, encode, read_token

logger = logging.getLogger('fake-smile')

MODEL: str = 'data/models/scrfd_10g_bnkps.onnx'

CHARGE_SECONDS: float = 3.0
WINDOW: float = 15.0
STREAM_FPS: float = 10.0


class FakeSmile:
    def __init__(self, link_url: str, token: str) -> None:
        self.link: Link = Link(link_url, token, self.on_message, self.on_frame, name='fake-smile')
        self.detector = None
        try:
            from modules.face import FaceDetector
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
        self.phone: str = ''

    def on_message(self, message: dict) -> None:
        kind: str = message.get('type', '')
        with self.mutex:
            if kind == 'line':
                self.waiting = int(message.get('waiting', 0))
            elif kind == 'turn':
                if message.get('id'):
                    self.phone = str(message['id'])
                    self.turn = True
                    self.turn_start = time.time()
                    self.charge = 0.0
                    self.charged_at = 0.0
                    logger.info('turn starts')
                self.asked = 0.0
            elif kind == 'gone':
                if self.turn and not self.busy:
                    self.turn = False
                    self.link.submit({'type': 'turn_end', 'reason': 'lost'})

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
        self.link.submit({'type': 'face', 'charge': round(charge, 3), 'faces': faces})
        if charge >= 1.0:
            # on its own thread: this one is the link's, which has to keep sending
            Thread(target=self.transform, args=(image,), daemon=True).start()

    def transform(self, image: np.ndarray) -> None:
        """The flash and a fake transformation streamed as the screen: the frame itself, worked over
        frame by frame, white at the flash, the blur falling, then the portrait standing; the stages as SMILE
        sends them (flash_in, flash_out, diffusion, grin, timeout), since the page streams from the flash-out on."""
        with self.mutex:
            if not self.turn or self.busy:
                return
            self.busy = True        # the turn goes on until the sequence is over: no judging, no new turn
        link: Link = self.link
        phone: str = self.phone
        link.submit({'type': 'stage', 'stage': 'flash', 'in': 0.9, 'out': 1.5})
        link.submit({'type': 'stage', 'stage': 'flash_in', 'progress': 1.0, 'value': 1.0})
        portrait: np.ndarray = cv2.applyColorMap(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_PINK)
        frames: int = int(6 * STREAM_FPS)
        for i in range(frames):
            t: float = i / frames
            if t < 0.15:
                shown: np.ndarray = cv2.addWeighted(image, t / 0.15, np.full_like(image, 255), 1 - t / 0.15, 0)
                stage: dict = {'type': 'stage', 'stage': 'flash_out', 'progress': round(t / 0.15, 3), 'value': round(t / 0.15, 3)}
            elif t < 0.7:
                blur: int = 2 * int(30 * (0.7 - t) / 0.55) + 1
                shown = cv2.GaussianBlur(image, (blur, blur), 0)
                stage = {'type': 'stage', 'stage': 'diffusion', 'progress': round((t - 0.15) / 0.55, 3)}
            else:
                shown = cv2.addWeighted(image, max(0.0, 1 - (t - 0.7) / 0.1), portrait, min(1.0, (t - 0.7) / 0.1), 0)
                stage = {'type': 'stage', 'stage': 'grin', 'progress': round((t - 0.7) / 0.3, 3)}
            link.submit(stage)
            link.submit_image(encode(shown) or b'')
            time.sleep(1.0 / STREAM_FPS)
        link.submit({'type': 'stage', 'stage': 'timeout', 'progress': 1.0})
        link.submit({'type': 'turn_end', 'reason': 'done'})
        with self.mutex:
            self.turn = False
            self.busy = False
        logger.info('turn ends (done)')
        # the delivery, after the turn: the files, then the result to the phone
        id: str = time.strftime('%Y_%m_%d_%H_%M_%S')
        path: str = os.path.join(tempfile.gettempdir(), f'fake_smile_{id}.jpg')
        cv2.imwrite(path, portrait)

        def done(ok: bool) -> None:
            link.submit({'type': 'result', 'id': id, 'phone': phone, 'name': 'Matthias Oostrik', 'caption': 'loves your smile!', 'ok': ok})
            os.remove(path)
            logger.info('result delivered (%s)', 'ok' if ok else 'upload failed')
        time.sleep(2.0)
        link.submit_upload([('portrait', id, path)], done)

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
                    self.link.submit({'type': 'turn_end', 'reason': 'missed'})
                    logger.info('turn missed')
                    turn = False
            self.link.submit({'type': 'eye', 'free': not turn, 'status': 'looking for faces'})
            if not turn and waiting > 0 and now - asked > 3.0:
                with self.mutex:
                    self.asked = now
                self.link.submit({'type': 'turn_start', 'window': WINDOW})


def main() -> None:
    parser = argparse.ArgumentParser(description='a stand-in for SMILE at the relay')
    parser.add_argument('--relay', default='ws://127.0.0.1:8800/ws/smile')
    parser.add_argument('--token-file', default='apps/smile/data/online_token.txt')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    token: str = read_token(args.token_file)
    if not token:
        parser.error(f'no token in {args.token_file}')
    FakeSmile(args.relay, token).run()


if __name__ == '__main__':
    main()
