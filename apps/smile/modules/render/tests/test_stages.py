"""Tests for the render's sequence to stage table and the sequencer's state changes (no GL needed)."""

import unittest

from modules.render.Sequencer import Sequencer, SequenceType
from modules.render.settings import SequencerSettings
from modules.render.stages import STAGES, CameraStage, DestructStage, DiffusionStage, FlashStage, RevealStage


class TestStages(unittest.TestCase):
    def test_every_sequence_type_has_a_stage(self):
        for seq in SequenceType:
            self.assertIn(seq, STAGES, seq.name)

    def test_cycle_stages(self):
        self.assertIs(STAGES[SequenceType.CAMERA_SHOW], CameraStage)
        self.assertIs(STAGES[SequenceType.CAPTURE], FlashStage)
        self.assertIs(STAGES[SequenceType.CONTROL_0], RevealStage)
        self.assertIs(STAGES[SequenceType.DIFF_LOOP], DiffusionStage)
        self.assertIs(STAGES[SequenceType.FADE], DestructStage)


class TestSequencer(unittest.TestCase):
    def setUp(self):
        self.sequencer = Sequencer(SequencerSettings())

    def tearDown(self):
        self.sequencer.close()

    def test_starts_only_from_camera_show(self):
        self.sequencer.start_sequence()
        self.assertEqual(self.sequencer.get_sequence(), SequenceType.CAMERA_DETECT)
        self.sequencer.set_sequence(SequenceType.DIFFUSION)
        self.sequencer.start_sequence()
        self.assertEqual(self.sequencer.get_sequence(), SequenceType.DIFFUSION)

    def test_update_reports_state_and_calls_back(self):
        messages = []
        self.sequencer.add_message_callback(messages.append)
        message = self.sequencer.update()
        self.assertEqual(message.sequence_type, SequenceType.CAMERA_SHOW)
        self.assertTrue(message.initial)
        self.assertEqual(messages, [message])
        self.assertIs(self.sequencer.message, message)
        self.assertEqual(self.sequencer.settings.state, 'CAMERA_SHOW')
