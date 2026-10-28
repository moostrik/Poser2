"""The saver puts a finished visitor in its style's folders, named by the time alone."""

import os
import re
import tempfile
import unittest

import json
import subprocess

import cv2
import numpy as np

from apps.smile.comfy.settings import PipelineSettings
from apps.smile.render.settings import SequencerSettings
from apps.smile.save import record
from apps.smile.save.ClipSaver import ClipSaver
from apps.smile.video import ffmpeg
from apps.smile.video.ffmpeg import Encoder


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

    def test_the_share_clip_has_the_track_and_the_watermark_and_the_players_clip_is_silent(self):
        settings = SequencerSettings()
        settings.encoder = Encoder.CPU
        settings.save = True
        settings.grin_fps = 10.0
        made = {'description': 'a face', 'seed': 7}
        with tempfile.TemporaryDirectory() as root:
            # a two second tone stands in for the track
            tone = os.path.join(root, 'tone.mp3')
            process = ffmpeg.popen(['-y', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2', '-q:a', '9', tone],
                                   subprocess.DEVNULL, subprocess.DEVNULL)
            process.communicate(timeout=30)
            self.assertEqual(process.returncode, 0)
            saved = []
            saver = ClipSaver(settings, root=root, watermark='Test Name')
            saver.set_saved_callback(lambda *args: saved.append(args))
            try:
                image = np.full((64, 64, 3), 128, np.uint8)
                saver.new_sequence(image, 'A')
                saver.set_record(made)
                for _ in range(15):
                    saver.save_frame(image)
                saver.set_share(tone, 0.5)
                saver.end_sequence(keep=True)
            finally:
                saver.stop()

            clip = os.path.join(root, 'grins', 'A', os.listdir(os.path.join(root, 'grins', 'A'))[0])
            shares = os.listdir(os.path.join(root, 'shares', 'A'))
            self.assertEqual(shares, [os.path.basename(clip)])
            shared = os.path.join(root, 'shares', 'A', shares[0])
            self.assertEqual(os.listdir(os.path.join(root, 'temp')), [])
            self.assertEqual(saved[0][2:], (True, clip, os.path.join(root, 'portraits', 'A', os.path.splitext(shares[0])[0] + '_portrait.jpg'), shared))

            def audio_streams(path):
                process = ffmpeg.popen(['-select_streams', 'a', '-show_entries', 'stream=codec_name,duration', '-of', 'json', path],
                                       subprocess.DEVNULL, subprocess.PIPE, command=ffmpeg.FFPROBE)
                out, _ = process.communicate(timeout=30)
                return json.loads(out)['streams']
            self.assertEqual(audio_streams(clip), [])                 # the players' clip is silent
            streams = audio_streams(shared)
            self.assertEqual(len(streams), 1)
            self.assertEqual(streams[0]['codec_name'], 'aac')
            self.assertAlmostEqual(float(streams[0]['duration']), 1.5, delta=0.3)   # the clip's length
            self.assertEqual(record.read(shared)['seed'], 7)           # the record travels with the share clip
            for path in (clip, shared):
                frames, fps = ffmpeg.decode_clip(path, (32, 40))
                self.assertEqual((len(frames), fps), (15, 10.0))
            # the watermark is in the share clip's picture, bottom right: brighter than the grey frame there
            frames, _ = ffmpeg.decode_clip(shared, (64, 64))
            self.assertGreater(int(frames[7, 8:40, 0:40].max()), 160)       # the mark, 24 px up; wider than the 64 px frame, so it reaches the left
            self.assertLess(int(frames[7, 0:12, :].max()), 140)             # nothing over the top

    def test_a_track_transcodes_to_aac_for_the_phones(self):
        with tempfile.TemporaryDirectory() as root:
            tone = os.path.join(root, 'tone.mp3')
            process = ffmpeg.popen(['-y', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1', '-q:a', '9', tone],
                                   subprocess.DEVNULL, subprocess.DEVNULL)
            process.communicate(timeout=30)
            out = os.path.join(root, 'tone.m4a')
            ffmpeg.transcode_aac(tone, out)
            process = ffmpeg.popen(['-select_streams', 'a:0', '-show_entries', 'stream=codec_name', '-of', 'json', out],
                                   subprocess.DEVNULL, subprocess.PIPE, command=ffmpeg.FFPROBE)
            streams = json.loads(process.communicate(timeout=30)[0])['streams']
            self.assertEqual(streams[0]['codec_name'], 'aac')
            with self.assertRaises(RuntimeError):
                ffmpeg.transcode_aac(os.path.join(root, 'missing.mp3'), out)

    def test_a_track_that_fails_leaves_the_players_clip_and_no_share_clip(self):
        settings = SequencerSettings()
        settings.encoder = Encoder.CPU
        settings.save = True
        with tempfile.TemporaryDirectory() as root:
            saved = []
            saver = ClipSaver(settings, root=root)
            saver.set_saved_callback(lambda *args: saved.append(args))
            try:
                image = np.full((64, 64, 3), 128, np.uint8)
                saver.new_sequence(image, 'A')
                for _ in range(5):
                    saver.save_frame(image)
                saver.set_share(os.path.join(root, 'missing.mp3'), 0.0)
                saver.end_sequence(keep=True)
            finally:
                saver.stop()
            clips = os.listdir(os.path.join(root, 'grins', 'A'))
            self.assertEqual(len(clips), 1)
            self.assertEqual(os.listdir(os.path.join(root, 'shares', 'A')), [])
            self.assertEqual(os.listdir(os.path.join(root, 'temp')), [])
            self.assertEqual(saved[0][5], '')
