"""Tests for the online turn: a fake link and camera, the sequence fed as messages, a clock passed in."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from modules.comfy.messages import GenMessage, GenState
from modules.comfy.settings import PipelineSettings
from modules.face.FaceFinder import FaceFrame
from modules.face.quality import FaceObservation, Pose, Rejection
from modules.online.Online import ASK_TIMEOUT, INSIDE, LOOKING, Online, Phase, core_description, face_message
from modules.online.settings import OnlineSettings
from modules.render.Sequencer import SeqenceMessage, SequenceType


class FakeLink:
    def __init__(self, url: str, token: str, on_message, on_frame) -> None:
        self.url, self.token = url, token
        self.on_message, self.on_frame = on_message, on_frame
        self.connected: bool = True
        self.started: bool = False
        self.stopped: bool = False
        self.sent: list[dict] = []
        self.images: list[bytes] = []
        self.uploads: list = []

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def send(self, message: dict) -> None:
        self.sent.append(message)

    def send_image(self, jpeg: bytes) -> None:
        self.images.append(jpeg)

    def upload(self, files, done) -> None:
        self.uploads.append((files, done))

    def types(self) -> list[str]:
        return [m['type'] for m in self.sent]

    def last(self, type: str) -> dict | None:
        return next((m for m in reversed(self.sent) if m['type'] == type), None)


class FakeCamera:
    def __init__(self) -> None:
        self.online: bool = False
        self.pushed: int = 0

    def set_online(self, active: bool) -> None:
        self.online = active

    def push_online(self, frame) -> None:
        self.pushed += 1


def observation(distance: float = 0.0, rejection: Rejection | None = None, subject: bool = True) -> FaceObservation:
    return FaceObservation(bbox=(0.3, 0.3, 0.7, 0.7), crop=(0.2, 0.1, 0.8, 0.85), kps=(), score=0.9, eye_dist=0.1, edge=0.1,
                           pose=Pose(0.0, 0.0, 0.0, 0.5), eyes=0.3, rejection=rejection, limit=20.0 if rejection else 0.0,
                           distance=distance, subject=subject)


class TestOnline(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        token = os.path.join(self.folder.name, 'token.txt')
        with open(token, 'w') as f:
            f.write('secret\n')
        self.settings = OnlineSettings()
        self.settings.token_file = token
        self.settings.turn_window = 10.0
        self.settings.eye_free_after = 2.0
        self.pipeline = PipelineSettings()
        self.camera = FakeCamera()
        self.sequencer = SimpleNamespace(settings=SimpleNamespace(pause=False))
        self.face = mock.Mock()
        self.links: list[FakeLink] = []

        def factory(url, token, on_message, on_frame):
            link = FakeLink(url, token, on_message, on_frame)
            self.links.append(link)
            return link
        self.online = Online(self.settings, self.pipeline, self.camera, self.sequencer, self.face,
                             record=lambda: {'description': 'a face'}, caption=lambda: ('Name', 'line'), link_factory=factory)
        self.now: float = 1000.0

    @property
    def link(self) -> FakeLink:
        return self.links[-1]

    def step(self, seq: SequenceType = SequenceType.CAMERA_SHOW, dt: float = 0.1, progress: float = 0.0, initial: bool = False,
             ready: bool = True, busy: bool = False) -> None:
        self.now += dt
        self.online.update(self.now, SeqenceMessage(seq, progress, initial), ready, busy)

    def go_idle(self) -> None:
        self.settings.active = True
        self.step()

    def start_turn(self) -> None:
        """The eye free long enough with someone waiting: the relay answers the turn."""
        self.go_idle()
        self.link.on_message({'type': 'line', 'waiting': 1})
        self.step(dt=self.settings.eye_free_after + 0.1)
        self.assertEqual(self.online.phase, Phase.ASKING)
        self.link.on_message({'type': 'turn', 'id': 'phone1'})
        self.step()
        self.assertEqual(self.online.phase, Phase.JUDGING)

    # OFF AND ON
    def test_off_does_nothing(self):
        self.step()
        self.assertEqual(self.links, [])
        self.assertEqual(self.settings.status, 'off')

    def test_no_token_is_a_status_and_no_link(self):
        self.settings.token_file = os.path.join(self.folder.name, 'missing.txt')
        self.go_idle()
        self.assertEqual(self.links, [])
        self.assertIn('no token', self.settings.status)

    def test_on_starts_the_link_and_tells_the_eye(self):
        self.go_idle()
        self.assertTrue(self.link.started)
        self.assertEqual(self.link.token, 'secret')
        eye = self.link.last('eye')
        self.assertEqual((eye['free'], eye['status']), (False, LOOKING))    # free only after eye_free_after
        self.step(dt=self.settings.eye_free_after + 0.1)
        self.assertTrue(self.link.last('eye')['free'])

    def test_the_eye_is_not_free_while_the_room_has_someone(self):
        self.go_idle()
        self.step(seq=SequenceType.CAMERA_DETECT, progress=0.3, dt=5.0)
        eye = self.link.last('eye')
        self.assertEqual((eye['free'], eye['status']), (False, INSIDE))
        self.step(seq=SequenceType.CAMERA_SHOW, dt=1.0)
        self.assertFalse(self.link.last('eye')['free'])      # free again only after the wait
        self.step(dt=2.0)
        self.assertTrue(self.link.last('eye')['free'])

    def test_off_during_a_turn_ends_it_and_the_link(self):
        self.start_turn()
        self.settings.active = False
        self.step()
        self.assertEqual(self.link.last('turn_end')['reason'], 'reset')
        self.assertTrue(self.link.stopped)
        self.assertFalse(self.camera.online)
        self.assertEqual(self.online.phase, Phase.OFF)

    # THE TURN
    def test_a_turn_starts_when_the_eye_is_free_and_someone_waits(self):
        self.start_turn()
        self.assertEqual(self.link.last('turn_start')['window'], 10.0)
        self.assertTrue(self.camera.online)
        self.face.reset.assert_called_once()
        self.assertEqual(self.settings.turns, 1)
        self.assertTrue(self.online.turn)

    def test_nobody_in_line_after_all_goes_back_to_idle(self):
        self.go_idle()
        self.link.on_message({'type': 'line', 'waiting': 1})
        self.step(dt=3.0)
        self.link.on_message({'type': 'turn', 'id': None})
        self.step()
        self.assertEqual(self.online.phase, Phase.IDLE)
        self.assertFalse(self.camera.online)

    def test_a_relay_that_does_not_answer_goes_back_to_idle(self):
        self.go_idle()
        self.link.on_message({'type': 'line', 'waiting': 1})
        self.step(dt=3.0)
        self.step(dt=ASK_TIMEOUT + 0.1)
        self.assertEqual(self.online.phase, Phase.IDLE)

    def test_frames_reach_the_camera_only_during_the_judging(self):
        self.go_idle()
        self.link.on_frame(b'not a jpeg')
        self.assertEqual(self.camera.pushed, 0)
        self.start_turn()
        import cv2
        ok, jpeg = cv2.imencode('.jpg', np.zeros((80, 64, 3), np.uint8))
        self.link.on_frame(jpeg.tobytes())
        self.assertEqual(self.camera.pushed, 1)

    def test_the_judging_goes_to_the_phone_with_the_charge(self):
        self.start_turn()
        self.step(seq=SequenceType.CAMERA_DETECT, progress=0.4)
        frame = FaceFrame((observation(0.2, Rejection.TURNED), observation(0.0, None, subject=False)), self.now)
        self.online.on_face(frame)
        face = self.link.last('face')
        self.assertEqual(face['charge'], 0.4)
        self.assertEqual(face['faces'][0]['reject'], 'yaw 0 > 20')
        self.assertEqual(face['faces'][0]['box'], [0.2, 0.1, 0.8, 0.85])
        self.assertIsNone(face['faces'][1]['reject'])

    def test_a_turn_without_a_face_is_missed_after_the_window(self):
        self.start_turn()
        self.step(dt=9.0)
        self.assertEqual(self.online.phase, Phase.JUDGING)
        self.step(dt=2.0)
        self.assertEqual(self.link.last('turn_end')['reason'], 'missed')
        self.assertFalse(self.camera.online)
        self.assertEqual(self.online.phase, Phase.IDLE)

    def test_a_charge_that_is_rising_at_the_window_keeps_the_turn(self):
        self.start_turn()
        self.step(seq=SequenceType.CAMERA_DETECT, progress=0.5, dt=11.0)
        self.assertEqual(self.online.phase, Phase.JUDGING)
        self.step(seq=SequenceType.CAMERA_SHOW, dt=1.0)
        self.assertEqual(self.online.phase, Phase.IDLE)

    def test_a_phone_that_leaves_loses_the_turn(self):
        self.start_turn()
        self.link.on_message({'type': 'gone'})
        self.step()
        self.assertEqual(self.link.last('turn_end')['reason'], 'lost')
        self.assertFalse(self.camera.online)

    def test_a_relay_that_drops_ends_the_turn_silently(self):
        self.start_turn()
        self.link.connected = False
        self.step()
        self.assertIsNone(self.link.last('turn_end'))
        self.assertFalse(self.camera.online)
        self.assertEqual(self.online.phase, Phase.IDLE)

    def test_the_flash_through_the_result(self):
        self.start_turn()
        self.step(seq=SequenceType.FLASH, initial=True)
        self.assertEqual(self.link.last('stage')['stage'], 'flash')
        self.assertEqual(self.online.phase, Phase.TRANSFORMING)
        self.assertTrue(self.camera.online)
        self.step(seq=SequenceType.CAPTURE)
        self.step(seq=SequenceType.ASK)
        self.assertFalse(self.camera.online)     # the capture is taken: the room's camera is back
        # the transformation's messages go to the phone
        image = np.zeros((100, 80, 3), np.uint8)
        self.online.on_generation(GenMessage(GenState.CAM_DEPTH, image=image))
        self.assertEqual(self.link.last('stage')['stage'], 'depth')
        self.assertEqual(len(self.link.images), 1)
        self.online.on_generation(GenMessage(GenState.DIFFUSION, 0.5, image=image))
        self.assertEqual([m['stage'] for m in self.link.sent if m['type'] == 'stage'][-2:], ['describe', 'diffusion'])
        self.assertEqual(self.link.last('stage')['progress'], 0.5)
        self.online.on_generation(GenMessage(GenState.DIFFUSION, 1.0, image=image, final=True))
        self.assertEqual(self.link.last('stage')['stage'], 'portrait')
        self.assertEqual(len(self.link.images), 2)      # the previews are rate limited, the portrait always goes
        self.online.on_generation(GenMessage(GenState.DIFF_LOOP, 0.1, image=image))
        self.assertEqual(self.link.last('stage')['stage'], 'grin')
        # the result stands: the saver closes the files, they are uploaded, the phone gets the result
        self.step(seq=SequenceType.TIMEOUT, initial=True)
        self.assertEqual(self.online.phase, Phase.DELIVERING)
        self.online.on_saved('v1', 'A', True, 'grins/A/v1.mp4', 'portraits/A/v1_portrait.jpg')
        self.step(seq=SequenceType.TIMEOUT)
        files, done = self.link.uploads[0]
        self.assertEqual([f[0] for f in files], ['portrait', 'still', 'grin'])
        self.assertEqual(files[2], ('grin', 'v1', 'grins/A/v1.mp4'))
        done(True)
        self.step(seq=SequenceType.RESET)       # the natural end of the visitor: not an operator reset
        result = self.link.last('result')
        self.assertEqual((result['id'], result['name'], result['caption']), ('v1', 'Name', 'line'))
        self.assertEqual(self.link.last('turn_end')['reason'], 'done')
        self.assertEqual(self.online.phase, Phase.IDLE)

    def test_the_description_is_qwens_words_only(self):
        self.pipeline.style_a.prefix = 'portrait of'
        self.pipeline.style_a.suffix = ', (serious: 1.2)'
        self.pipeline.style_current = 'A'
        online = Online(self.settings, self.pipeline, self.camera, self.sequencer, self.face,
                        record=lambda: {'description': 'portrait of a woman, red hair , (serious: 1.2)'}, caption=lambda: ('', ''),
                        link_factory=FakeLink)
        self.assertEqual(online.description(), 'a woman, red hair')
        self.assertEqual(core_description('plain', '', ''), 'plain')

    def test_a_reset_during_the_transformation_ends_the_turn(self):
        self.start_turn()
        self.step(seq=SequenceType.FLASH, initial=True)
        self.step(seq=SequenceType.RESET)
        self.assertEqual(self.link.last('turn_end')['reason'], 'reset')
        self.assertFalse(self.camera.online)

    def test_a_visitor_that_was_not_kept_has_no_result(self):
        self.start_turn()
        self.step(seq=SequenceType.FLASH, initial=True)
        self.step(seq=SequenceType.TIMEOUT, initial=True)
        self.online.on_saved('v1', 'A', False, '', '')
        self.step(seq=SequenceType.TIMEOUT)
        self.assertEqual(self.link.last('turn_end')['reason'], 'reset')
        self.assertIsNone(self.link.last('result'))

    def test_a_failed_upload_has_no_result(self):
        self.start_turn()
        self.step(seq=SequenceType.FLASH, initial=True)
        self.step(seq=SequenceType.TIMEOUT, initial=True)
        self.online.on_saved('v1', 'A', True, 'g.mp4', 'p.jpg')
        self.step(seq=SequenceType.TIMEOUT)
        self.link.uploads[0][1](False)
        self.step(seq=SequenceType.TIMEOUT)
        self.assertEqual(self.link.last('turn_end')['reason'], 'reset')

    def test_face_message_shape(self):
        frame = FaceFrame((observation(0.5),), 1.0)
        message = face_message(frame, 1.0)
        self.assertEqual(message['faces'][0]['distance'], 0.5)
        self.assertTrue(message['faces'][0]['subject'])


if __name__ == '__main__':
    unittest.main()
