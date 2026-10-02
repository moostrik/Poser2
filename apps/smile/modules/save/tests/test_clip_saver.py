"""The saver puts a finished visitor in its style's folders, named by the time alone."""

import os
import re
import tempfile
import unittest

import cv2
import numpy as np

from modules.comfy.settings import PipelineSettings
from modules.render.settings import SequencerSettings
from modules.save import record
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
            # no record was handed over: none in the files
            self.assertIsNone(record.read(os.path.join(root, 'grins', 'B', clips[0])))
            self.assertIsNone(record.read(os.path.join(root, 'portraits', 'B', id + '_portrait.jpg')))

    def test_the_record_is_in_the_clip_and_the_portrait(self):
        settings = SequencerSettings()
        settings.encoder = Encoder.CPU
        settings.save = True
        made = {'description': 'a face, "quoted"', 'seed': 7, 'grin': {'smile': 2.0},
                'settings': {'pipeline': PipelineSettings().to_dict()}}
        with tempfile.TemporaryDirectory() as root:
            saver = ClipSaver(settings, root=root)
            try:
                image = np.full((64, 64, 3), 128, np.uint8)
                saver.new_sequence(image, 'A')
                saver.set_record(made)
                saver.save_portrait(image)
                for _ in range(5):
                    saver.save_frame(image)
                saver.end_sequence(keep=True)
            finally:
                saver.stop()

            clip = os.listdir(os.path.join(root, 'grins', 'A'))[0]
            id = os.path.splitext(clip)[0]
            expected = {'id': id, 'style': 'A', **made}
            self.assertEqual(record.read(os.path.join(root, 'grins', 'A', clip)), expected)
            self.assertEqual(record.read(os.path.join(root, 'portraits', 'A', id + '_portrait.jpg')), expected)
            self.assertIsNotNone(cv2.imread(os.path.join(root, 'portraits', 'A', id + '_portrait.jpg')))
