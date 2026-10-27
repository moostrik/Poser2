"""Tests for the online turn: a fake link, camera and screen grab, the sequence fed as messages, a clock passed in."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from modules.face.FaceFinder import FaceFrame
from modules.face.quality import FaceObservation, Pose, Rejection
from modules.online.Online import ASK_TIMEOUT, INSIDE, LOOKING, SAVE_TIMEOUT, Online, Phase, face_message
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
        self.frames: list = []
        self.uploads: list = []

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def send(self, message: dict) -> None:
        self.sent.append(message)

    def send_frame(self, frame) -> None:
        self.frames.append(frame)

    def upload(self, files, done) -> None:
        self.uploads.append((files, done))

    def types(self) -> list[str]:
        return [m['type'] for m in self.sent]

    def last(self, type: str) -> dict | None:
        return next((m for m in reversed(self.sent) if m['type'] == type), None)


class FakeGrab:
    """The render's screen grab: frames are put in by the test, taken once."""
    def __init__(self) -> None:
        self.on: bool = False
        self.width: int = 0
        self.fps: float = 0.0
        self.frame = None

    def set(self, on: bool, width: int = 512, fps: float = 10.0) -> None:
        self.on, self.width, self.fps = on, width, fps

    def take(self):
        frame, self.frame = self.frame, None
        return frame


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
        self.camera = FakeCamera()
        self.grab = FakeGrab()
        self.sequencer = SimpleNamespace(settings=SimpleNamespace(pause=False, flash_in=0.5, flash_out=1.25))
        self.face = mock.Mock()
        self.links: list[FakeLink] = []

        def factory(url, token, on_message, on_frame):
            link = FakeLink(url, token, on_message, on_frame)
            self.links.append(link)
            return link
        self.music: tuple = ('none', None)      # the visitor's track: none unless a test gives one
        self.laughs: list = []                  # the laughter samples SMILE has
        self.laughter: bool = True              # the panel's laughter switch
        self.picks: int = 0                     # Main.pick_style calls
        self.online = Online(self.settings, self.camera, self.sequencer, self.face, self.grab,
                             caption=lambda: ('Name', 'line'), music=lambda: self.music,
                             laughs=lambda: self.laughs, laughter=lambda: self.laughter, pick=self.pick, link_factory=factory)
        self.now: float = 1000.0

    @property
    def link(self) -> FakeLink:
        return self.links[-1]

    def pick(self) -> None:
        self.picks += 1

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

    def test_the_flash_streams_the_screen_until_the_sequence_is_over(self):
        self.settings.stream_width = 320
        self.settings.stream_fps = 12.0
        self.start_turn()
        self.assertFalse(self.grab.on)
        self.step(seq=SequenceType.FLASH, initial=True)
        stages = lambda: [m for m in self.link.sent if m['type'] == 'stage']
        self.assertEqual(stages()[0], {'type': 'stage', 'stage': 'flash', 'in': 0.5, 'out': 1.25})
        self.assertEqual(stages()[1], {'type': 'stage', 'stage': 'flash_in', 'progress': 0.0})
        self.assertEqual(self.online.phase, Phase.TRANSFORMING)
        self.assertTrue(self.camera.online)
        self.assertEqual((self.grab.on, self.grab.width, self.grab.fps), (True, 320, 12.0))
        # the screen's frames go to the phone as the render grabs them
        frame = np.zeros((400, 320, 4), np.uint8)
        self.grab.frame = frame
        self.step(seq=SequenceType.CAPTURE)
        self.assertIs(self.link.frames[0], frame)
        self.step(seq=SequenceType.ASK)
        self.assertEqual(len(self.link.frames), 1)      # nothing new, nothing sent
        self.assertFalse(self.camera.online)     # the capture is taken: the room's camera is back
        self.assertEqual(len(stages()), 2)       # the same stage at the same progress: not sent again
        # the stages with their progress, a few times a second, for the phone's tape
        self.step(seq=SequenceType.DIFFUSION, progress=0.5)
        self.assertEqual(self.link.last('stage'), {'type': 'stage', 'stage': 'diffusion', 'progress': 0.5})
        self.step(seq=SequenceType.DIFFUSION, progress=0.6, dt=0.05)
        self.assertEqual(self.link.last('stage')['progress'], 0.5)      # too soon
        self.step(seq=SequenceType.DIFFUSION, progress=0.7, dt=0.3)
        self.assertEqual(self.link.last('stage')['progress'], 0.7)
        self.step(seq=SequenceType.DIFF_LOOP)
        self.assertEqual(self.link.last('stage')['stage'], 'grin')
        self.step(seq=SequenceType.TIMEOUT, initial=True)
        self.assertEqual(self.online.phase, Phase.TRANSFORMING)
        self.assertEqual((self.grab.on, self.grab.fps), (True, 2.0))      # the result stands: the link is the clip's
        self.assertIsNone(self.link.last('turn_end'))
        # the sequencer passes RESET at the end of the timeout, then the bloom and the fade: the turn
        # goes on, the picture through the bloom, the sound through the fade
        self.step(seq=SequenceType.RESET)
        self.assertEqual(self.online.phase, Phase.TRANSFORMING)
        self.step(seq=SequenceType.BLOOM, progress=0.5)
        self.assertEqual(self.link.last('stage'), {'type': 'stage', 'stage': 'bloom', 'progress': 0.5})
        self.assertTrue(self.grab.on)
        self.step(seq=SequenceType.FADE, progress=0.2)
        self.assertEqual(self.link.last('stage')['stage'], 'fade')
        self.assertFalse(self.grab.on)
        self.assertIsNone(self.link.last('turn_end'))
        # back at the camera: the turn ends, no result yet
        self.step(seq=SequenceType.CAMERA_SHOW)
        self.assertEqual(self.link.last('turn_end')['reason'], 'done')
        self.assertEqual(self.online.phase, Phase.IDLE)
        self.assertIsNone(self.link.last('result'))

    # THE MUSIC
    def test_the_track_is_picked_and_named_at_the_turns_start(self):
        self.music = ('0001', 'data/audio/SMILE-BS-Music/0001 V030 CC.mp3')
        self.go_idle()
        self.link.on_message({'type': 'library', 'tracks': ['0001'], 'laughs': []})
        self.start_turn()
        self.assertEqual(self.picks, 1)
        self.assertEqual(self.link.last('music'), {'type': 'music', 'id': '0001', 'url': '/files/tracks/0001.m4a', 'offset': 0.0})
        self.step(seq=SequenceType.CAMERA_DETECT, progress=0.4)
        self.step(seq=SequenceType.FLASH, initial=True)
        self.assertEqual(len([m for m in self.link.sent if m['type'] == 'music']), 1)    # once per turn

    def test_a_track_the_relay_lacks_is_uploaded_first(self):
        self.music = ('0001', 'data/audio/SMILE-BS-Music/0001 V030 CC.mp3')
        self.start_turn()
        self.step(seq=SequenceType.FLASH, initial=True)
        self.assertIsNone(self.link.last('music'))
        files, done = self.link.uploads[0]
        self.assertEqual(files, [('track', '0001', 'data/audio/SMILE-BS-Music/0001 V030 CC.mp3')])
        self.step(seq=SequenceType.CAPTURE, dt=1.2)
        done(True)
        self.step(seq=SequenceType.ASK, dt=0.1)
        music = self.link.last('music')
        self.assertEqual((music['id'], music['offset']), ('0001', 1.3))     # the seconds since the flash
        # the relay has it now: the next visitor's goes at once
        self.step(seq=SequenceType.TIMEOUT, initial=True)
        self.step(seq=SequenceType.RESET)
        self.step(seq=SequenceType.CAMERA_SHOW)     # the turn ends here
        self.start_turn()
        self.step(seq=SequenceType.FLASH, initial=True)
        self.assertEqual(self.link.last('music')['offset'], 0.0)
        self.assertEqual(len(self.link.uploads), 1)

    def test_a_track_uploaded_after_its_turn_is_not_sent(self):
        self.music = ('0001', 'x.mp3')
        self.start_turn()
        self.step(seq=SequenceType.FLASH, initial=True)
        _, done = self.link.uploads[0]
        self.step(seq=SequenceType.RESET)       # the operator's reset
        done(True)
        self.step()
        self.assertIsNone(self.link.last('music'))

    def test_no_music_without_a_track_or_with_the_switch_off(self):
        self.start_turn()
        self.step(seq=SequenceType.FLASH, initial=True)
        self.assertIsNone(self.link.last('music'))
        self.assertEqual(self.link.uploads, [])
        self.step(seq=SequenceType.RESET)
        self.settings.music = False
        self.music = ('0001', 'x.mp3')
        self.link.on_message({'type': 'library', 'tracks': ['0001'], 'laughs': []})
        self.start_turn()
        self.step(seq=SequenceType.FLASH, initial=True)
        self.assertIsNone(self.link.last('music'))

    def test_the_laughter_samples_the_relay_lacks_are_uploaded_once(self):
        self.laughs = [('a.mp3', 'l/a.mp3'), ('b.wav', 'l/b.wav'), ('c.mp3', 'l/c.mp3')]
        self.go_idle()
        self.link.on_message({'type': 'library', 'tracks': [], 'laughs': ['a.mp3']})
        self.step()
        self.assertEqual(self.link.uploads[0][0], [('laugh', 'b', 'l/b.wav'), ('laugh', 'c', 'l/c.mp3')])
        self.link.on_message({'type': 'library', 'tracks': [], 'laughs': ['a.mp3', 'b.wav', 'c.mp3']})
        self.step()
        self.assertEqual(len(self.link.uploads), 1)

    def test_the_eye_carries_the_laughter_switch(self):
        self.go_idle()
        self.assertTrue(self.link.last('eye')['laughter'])
        self.laughter = False
        self.step()
        self.assertFalse(self.link.last('eye')['laughter'])     # a change goes at once

    def test_the_result_is_delivered_as_soon_as_the_files_are_there(self):
        self.start_turn()
        self.step(seq=SequenceType.FLASH, initial=True)
        self.step(seq=SequenceType.ASK)
        # the saver closes the files during the timeout: the upload starts then, before the turn is over
        self.step(seq=SequenceType.TIMEOUT, initial=True)
        self.online.on_saved('v1', 'A', True, 'grins/A/v1.mp4', 'portraits/A/v1_portrait.jpg')
        self.step(seq=SequenceType.TIMEOUT)
        files, done = self.link.uploads[0]
        self.assertEqual([f[0] for f in files], ['portrait', 'still', 'grin'])
        self.assertEqual(files[2], ('grin', 'v1', 'grins/A/v1.mp4'))     # no share clip: the players' clip
        done(True)
        result = self.link.last('result')
        self.assertEqual((result['id'], result['phone'], result['name'], result['caption'], result['ok']),
                         ('v1', 'phone1', 'Name', 'line', True))
        self.assertIsNone(self.link.last('turn_end'))                   # the turn goes on
        self.step(seq=SequenceType.CAMERA_SHOW)
        self.assertEqual(self.link.last('turn_end')['reason'], 'done')
        self.step(seq=SequenceType.CAMERA_SHOW)
        self.assertEqual(len(self.link.uploads), 1)     # once

    def test_the_files_may_come_after_the_turn_and_the_eye_is_free_meanwhile(self):
        self.start_turn()
        self.step(seq=SequenceType.FLASH, initial=True)
        self.step(seq=SequenceType.TIMEOUT, initial=True)
        self.step(seq=SequenceType.CAMERA_SHOW)
        self.assertEqual(self.link.last('turn_end')['reason'], 'done')
        self.step()                                         # idle: the free count starts
        self.step(dt=self.settings.eye_free_after + 0.1)
        self.assertTrue(self.link.last('eye')['free'])      # the delivery does not hold the eye
        self.online.on_saved('v1', 'A', True, 'g.mp4', 'p.jpg', 'shares/A/v1.mp4')
        self.step()
        self.assertEqual(len(self.link.uploads), 1)
        self.assertEqual(self.link.uploads[0][0][2], ('grin', 'v1', 'shares/A/v1.mp4'))   # the phone's video is the share clip
        self.link.uploads[0][1](False)
        self.assertFalse(self.link.last('result')['ok'])

    def test_no_files_within_the_save_timeout_is_a_result_without_them(self):
        self.start_turn()
        self.step(seq=SequenceType.FLASH, initial=True)
        self.step(seq=SequenceType.TIMEOUT, initial=True)
        self.step(seq=SequenceType.CAMERA_SHOW)
        self.step(dt=SAVE_TIMEOUT + 1)
        result = self.link.last('result')
        self.assertEqual((result['id'], result['phone'], result['ok']), ('', 'phone1', False))
        self.assertEqual(self.link.uploads, [])

    def test_a_reset_during_the_transformation_ends_the_turn_and_drops_the_delivery(self):
        self.start_turn()
        self.step(seq=SequenceType.FLASH, initial=True)
        self.step(seq=SequenceType.DIFFUSION)
        self.step(seq=SequenceType.RESET)       # before the timeout: the operator's
        self.assertEqual(self.link.last('turn_end')['reason'], 'reset')
        self.assertFalse(self.camera.online)
        self.assertFalse(self.grab.on)
        self.online.on_saved('v1', 'A', False, '', '')
        self.step(seq=SequenceType.CAMERA_SHOW, dt=SAVE_TIMEOUT + 1)
        self.assertIsNone(self.link.last('result'))

    def test_a_visitor_that_was_not_kept_is_a_result_without_files(self):
        self.start_turn()
        self.step(seq=SequenceType.FLASH, initial=True)
        self.step(seq=SequenceType.TIMEOUT, initial=True)
        self.online.on_saved('v1', 'A', False, '', '')
        self.step(seq=SequenceType.CAMERA_SHOW)
        self.step(seq=SequenceType.CAMERA_SHOW)
        self.assertEqual(self.link.last('turn_end')['reason'], 'done')
        result = self.link.last('result')
        self.assertEqual((result['id'], result['ok']), ('v1', False))
        self.assertEqual(self.link.uploads, [])

    def test_face_message_shape(self):
        frame = FaceFrame((observation(0.5),), 1.0)
        message = face_message(frame, 1.0)
        self.assertEqual(message['faces'][0]['distance'], 0.5)
        self.assertTrue(message['faces'][0]['subject'])


if __name__ == '__main__':
    unittest.main()
