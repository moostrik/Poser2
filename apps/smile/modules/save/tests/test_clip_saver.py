"""The saver puts a finished visitor in its style's folders, named by the time alone."""

import os
import re
import tempfile
import unittest

import numpy as np

from modules.render.settings import SequencerSettings
from modules.save.ClipSaver import ClipSaver
from modules.video import ffmpeg
from modules.video.ffmpeg import Encoder


@unittest.skipUnless(ffmpeg.available(), 'ffmpeg not on the PATH')
class TestClipSaverStyle(unittest.TestCase):

    def test_a_kept_visitor_lands_in_its_style_folders(self):
        settings = SequencerSettings()
        settings.encoder = Encoder.CPU
        settings.save = True
        with tempfile.TemporaryDirectory() as root:
            saver = ClipSaver(settings, root=root)
            try:
                image = np.full((64, 64, 3), 128, np.uint8)
                saver.new_sequence(image, 'B')
                saver.save_portrait(image)
                for _ in range(5):
                    saver.save_frame(image)
                saver.end_sequence(keep=True)
            finally:
                saver.stop()

            clips = os.listdir(os.path.join(root, 'grins', 'B'))
            self.assertEqual(len(clips), 1)
            self.assertRegex(clips[0], r'^\d{4}(_\d{2}){5}\.mp4$')
            id = os.path.splitext(clips[0])[0]
            self.assertEqual(sorted(os.listdir(os.path.join(root, 'portraits', 'B'))),
                             [id + '_capture.jpg', id + '_portrait.jpg'])
            self.assertEqual(os.listdir(os.path.join(root, 'grins', 'A')), [])
            self.assertEqual(os.listdir(os.path.join(root, 'temp')), [])
            self.assertFalse(re.search(r'_[AB]$', id))
