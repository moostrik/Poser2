"""Tests for what the app sends to the Max patch every frame and for the settings tree the Audio
tab is built from (no camera, no GL, no ComfyUI: the callback runs on a stand-in for the app)."""

import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from apps.smile.main import Main
from apps.smile.settings import Settings
from apps.smile.osc.settings import OscSettings
from apps.smile.render.Sequencer import SeqenceMessage, SequenceType
from modules.settings import presets


class TestAudioTree(unittest.TestCase):
    def test_osc_lives_under_audio(self):
        settings = Settings()
        self.assertIsInstance(settings.audio.osc, OscSettings)
        self.assertNotIn('osc', settings.children)
        self.assertIn('osc', settings.to_dict()['audio'])

    def test_every_preset_in_the_repo_loads_without_a_warning(self):
        # a preset with the OSC block still at the root would log 'ignoring unknown key'
        paths = [p for p in sorted(Path('apps/smile/data/settings').glob('*.json')) if not p.name.startswith('.')]
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
            audio=mock.Mock(get_audio_id=lambda: '0001'), osc=mock.Mock(), screen_test=mock.Mock(),
            _limit_visitor=lambda seq: None,
            _style_picked=False, _face_frame_time=0.0, _visitor_start=None, _saving=False, _grin_saved=False,
            credit_scroller=mock.Mock(), online=mock.Mock(showing=False), clip_player=mock.Mock())
        self.app._online_cell = lambda: Main._online_cell(self.app)

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
        self.assertEqual(sent['/laughter_offset'], 0.0)
        audio = self.settings.audio
        audio.sound, audio.laughter = False, False
        audio.music_volume, audio.laughter_volume, audio.laughter_offset = -12.0, -6.0, -90.0
        sent = self.frame()
        self.assertEqual((sent['/sound'], sent['/laughter']), (0, 0))
        self.assertEqual((sent['/music_volume'], sent['/laughter_volume']), (-12.0, -6.0))
        self.assertEqual(sent['/laughter_offset'], -90.0)

    def test_the_switches_go_out_as_ints_and_the_levels_as_floats(self):
        # True == 1, so the values alone would pass a bool, which python-osc sends as OSC True
        sent = self.frame()
        for address in ('/sound', '/laughter', '/laughter_test'):
            self.assertIs(type(sent[address]), int, address)
        for address in ('/music_volume', '/laughter_volume', '/laughter_offset'):
            self.assertIs(type(sent[address]), float, address)

    def test_nothing_goes_out_while_osc_is_off(self):
        self.settings.audio.osc.active = False
        self.assertEqual(self.frame(), {})

    def test_the_laughter_test_runs_every_frame_holds_new_visitors_and_tells_max(self):
        sent = self.frame()
        self.app.screen_test.update.assert_called_once()
        self.assertFalse(self.app.sequencer.blocked)
        self.assertEqual((sent['/laughter_test'], sent['/laughter']), (0, 1))
        self.settings.audio.laughter_test = True
        sent = self.frame()
        self.assertTrue(self.app.sequencer.blocked)
        self.assertEqual((sent['/laughter_test'], sent['/laughter']), (1, 1))    # the panel's switch is untouched
        self.settings.audio.laughter_test = False
        sent = self.frame()
        self.assertFalse(self.app.sequencer.blocked)
        self.assertEqual(sent['/laughter_test'], 0)

    def test_the_test_bang_goes_to_max_with_the_screen_and_the_sample(self):
        Main._test_bang(self.app, 3, 1)
        self.app.osc.send.assert_called_once_with('/test_bang', (3, 1))

    def test_the_online_cell_follows_the_screen_setting_the_link_and_the_turn(self):
        online = self.settings.online
        online.active, online.screen = True, 27         # screen 27: wall 1, unit 1 output 3, cell 12
        online.connected = False
        self.frame()
        self.assertIsNone(self.app.clip_player.online_cell)
        self.app.render.set_online_cell.assert_called_with(1, 12, 'clip', 'http://127.0.0.1:8800/')
        online.connected = True
        self.frame()
        self.assertEqual(self.app.clip_player.online_cell, 24 + 12)     # the player's cells count over both walls
        self.app.render.set_online_cell.assert_called_with(1, 12, 'qr', 'http://127.0.0.1:8800/')
        self.app.online.showing = True
        online.page = ' https://smile.example/ '
        self.frame()
        self.app.render.set_online_cell.assert_called_with(1, 12, 'screen', 'https://smile.example/')
        online.screen = 0
        self.frame()
        self.assertIsNone(self.app.clip_player.online_cell)
        self.assertEqual(self.app.render.set_online_cell.call_args.args[2], 'clip')


if __name__ == '__main__':
    unittest.main()
