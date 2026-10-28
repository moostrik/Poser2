"""Tests for the laughter test stepper: recording the lights and bangs, a clock passed in, a folder of
empty sample files."""

import tempfile
import unittest
from pathlib import Path

from apps.smile.audio.ScreenTest import SCREENS, ScreenTest, list_samples
from apps.smile.audio.settings import AudioSettings

SAMPLES: tuple[str, ...] = ('Angry.mp3', 'v004_1.wav', 'v004_10.wav', 'V005_12.wav', 'v005_11.wav')


class TestListSamples(unittest.TestCase):
    def test_sound_files_in_the_folder_order_case_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in SAMPLES + ('notes.txt', '.hidden.wav'):
                Path(folder, name).touch()
            self.assertEqual(list_samples(folder), ['Angry.mp3', 'v004_1.wav', 'v004_10.wav', 'v005_11.wav', 'V005_12.wav'])

    def test_a_missing_folder_is_empty(self):
        self.assertEqual(list_samples('no/such/folder'), [])


class TestScreenTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        for name in SAMPLES:
            Path(self.folder.name, name).touch()
        self.settings = AudioSettings()
        self.sent: list[tuple[int, int, float, int]] = []
        self.banged: list[tuple[int, int]] = []
        self.test = self.make(self.settings)

    def make(self, settings: AudioSettings) -> ScreenTest:
        return ScreenTest(settings,
                          send=lambda wall, cell, seconds, number: self.sent.append((wall, cell, seconds, number)),
                          bang=lambda index, sample: self.banged.append((index, sample)),
                          folder=self.folder.name)

    def test_the_samples_are_offered_and_the_first_is_the_default(self):
        self.assertEqual(self.settings.laughter_samples, ['Angry.mp3', 'v004_1.wav', 'v004_10.wav', 'v005_11.wav', 'V005_12.wav'])
        self.assertEqual(self.settings.laughter_test_sample, 'Angry.mp3')

    def test_a_chosen_file_keeps_and_a_gone_one_falls_back(self):
        settings = AudioSettings()
        settings.laughter_test_sample = 'v005_11.wav'
        self.make(settings)
        self.assertEqual(settings.laughter_test_sample, 'v005_11.wav')
        settings.laughter_test_sample = 'gone.wav'
        self.make(settings)
        self.assertEqual(settings.laughter_test_sample, 'Angry.mp3')

    def test_the_bang_carries_the_files_number_in_the_patch(self):
        self.settings.laughter_test_sample = 'v005_11.wav'
        self.settings.laughter_test = True
        self.test.update(0.0)
        self.assertEqual(self.banged, [(0, 4)])
        self.settings.laughter_test_sample = 'V005_12.wav'
        self.test.update(2.0)
        self.assertEqual(self.banged[-1], (6, 5))     # screen 2 is cell 6

    def test_off_sends_nothing_and_shows_0(self):
        self.test.update(0.0)
        self.test.update(5.0)
        self.assertEqual(self.sent, [])
        self.assertEqual(self.banged, [])
        self.assertEqual(self.settings.laughter_test_screen, 0)

    def test_a_preset_saved_with_the_test_on_starts_off(self):
        settings = AudioSettings()
        settings.laughter_test = True
        settings.laughter_test_screen = 7
        self.make(settings)
        self.assertFalse(settings.laughter_test)
        self.assertEqual(settings.laughter_test_screen, 0)

    def test_steps_round_the_screens_one_wall_then_the_other(self):
        self.settings.laughter_test = True
        self.settings.laughter_test_step = 2.0
        self.test.update(10.0)      # the first screen at once
        self.assertEqual(self.sent, [(0, 0, 2.0, 1)])
        self.assertEqual(self.banged, [(0, 1)])
        self.assertEqual(self.settings.laughter_test_screen, 1)
        self.test.update(11.9)      # not yet
        self.assertEqual(len(self.sent), 1)
        self.test.update(12.0)
        self.assertEqual(self.sent[-1], (0, 6, 2.0, 2))         # screen 2 is cell 6: unit 1, output 2
        self.assertEqual(self.banged[-1], (6, 1))
        self.assertEqual(self.settings.laughter_test_screen, 2)
        for i in range(2, SCREENS):
            self.test.update(10.0 + 2.0 * i)
        self.assertEqual(self.sent[24], (1, 0, 2.0, 25))        # the second wall's first cell
        self.assertEqual(self.banged[24], (24, 1))
        self.assertEqual(self.sent[-1], (1, 23, 2.0, 48))
        self.assertEqual(self.settings.laughter_test_screen, SCREENS)
        self.test.update(10.0 + 2.0 * SCREENS)                  # round again
        self.assertEqual(self.sent[-1], (0, 0, 2.0, 1))
        self.assertEqual(self.settings.laughter_test_screen, 1)

    def test_off_then_on_starts_at_screen_1_again(self):
        self.settings.laughter_test = True
        self.test.update(0.0)
        self.test.update(2.0)
        self.settings.laughter_test = False
        self.test.update(3.0)
        self.assertEqual(self.settings.laughter_test_screen, 0)
        self.settings.laughter_test = True
        self.test.update(4.0)
        self.assertEqual(self.sent[-1], (0, 0, 2.0, 1))
        self.assertEqual(self.settings.laughter_test_screen, 1)

    def test_a_failing_send_or_bang_is_one_warning_and_the_test_goes_on(self):
        def fail(*args) -> None:
            raise OSError('port unreachable')
        for failing in ({'send': fail}, {'bang': fail}):
            with self.subTest(**failing):
                settings = AudioSettings()
                test = ScreenTest(settings, folder=self.folder.name, **failing)
                settings.laughter_test = True
                with self.assertLogs('apps.smile.audio.ScreenTest', 'WARNING') as logs:
                    test.update(0.0)
                    test.update(2.0)
                self.assertEqual(len(logs.records), 1)
                self.assertEqual(settings.laughter_test_screen, 2)


if __name__ == '__main__':
    unittest.main()
