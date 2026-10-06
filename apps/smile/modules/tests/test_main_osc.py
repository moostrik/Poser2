"""Tests for what the app sends to the Max patch every frame and for the settings tree the Audio
tab is built from (no camera, no GL, no ComfyUI: the callback runs on a stand-in for the app)."""

import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from modules.main import Main
from modules.main_settings import Settings
from modules.osc.settings import OscSettings
from modules.render.Sequencer import SeqenceMessage, SequenceType
from modules.settings import presets


class TestAudioTree(unittest.TestCase):
    def test_osc_lives_under_audio(self):
        settings = Settings()
        self.assertIsInstance(settings.audio.osc, OscSettings)
        self.assertNotIn('osc', settings.children)
        self.assertIn('osc', settings.to_dict()['audio'])

    def test_every_preset_in_the_repo_loads_without_a_warning(self):
        # a preset with the OSC block still at the root would log 'ignoring unknown key'
        paths = [p for p in sorted(Path('data/settings').glob('*.json')) if not p.name.startswith('.')]
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(preset=path.name), self.assertNoLogs('modules.settings', logging.WARNING):
                self.assertTrue(presets.load(Settings(), str(path)))


class TestMaxControls(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.app = SimpleNamespace(
            settings=self.settings, diffuse=mock.Mock(ready=True),
            # a Mock attribute is truthy: the camera must really not be stalled
            camera=SimpleNamespace(frame=lambda: (None, 0.0), stalled=False, next_image=mock.Mock()),
            render=mock.Mock(), artnet=mock.Mock(), face=mock.Mock(), save=mock.Mock(), sequencer=mock.Mock(),
            audio=mock.Mock(get_audio_id=lambda: '0001'), osc=mock.Mock(),
            _limit_visitor=lambda seq: None,
            _style_picked=False, _face_frame_time=0.0, _visitor_start=None, _saving=False, _grin_saved=False)

    def frame(self) -> dict:
        """One sequencer frame: what went to the OSC sender, by address."""
        self.app.osc.send.reset_mock()
        Main.sequence_callback(self.app, SeqenceMessage(SequenceType.CAMERA_SHOW, 0.0))
        return {c.args[0]: c.args[1] for c in self.app.osc.send.call_args_list}

    def test_the_panel_controls_go_out_every_frame_in_max_units(self):
        sent = self.frame()
        self.assertEqual(sent['/sound'], 1)
        self.assertEqual(sent['/music_volume'], 0.0)
        self.assertEqual(sent['/laughter'], 1)
        self.assertEqual(sent['/laughter_volume'], 0.0)
        audio = self.settings.audio
        audio.sound, audio.laughter = False, False
        audio.music_volume, audio.laughter_volume = -12.0, -6.0
        sent = self.frame()
        self.assertEqual((sent['/sound'], sent['/laughter']), (0, 0))
        self.assertEqual((sent['/music_volume'], sent['/laughter_volume']), (-12.0, -6.0))

    def test_the_switches_go_out_as_ints_and_the_levels_as_floats(self):
        # True == 1, so the values alone would pass a bool, which python-osc sends as OSC True
        sent = self.frame()
        for address in ('/sound', '/laughter'):
            self.assertIs(type(sent[address]), int, address)
        for address in ('/music_volume', '/laughter_volume'):
            self.assertIs(type(sent[address]), float, address)

    def test_nothing_goes_out_while_osc_is_off(self):
        self.settings.audio.osc.active = False
        self.assertEqual(self.frame(), {})


if __name__ == '__main__':
    unittest.main()
