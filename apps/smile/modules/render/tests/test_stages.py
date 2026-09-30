"""Tests for the render's sequence to stage table and the sequencer's state changes (no GL needed)."""

import unittest
from unittest import mock

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

    def flash_until_ask(self, clock: list[float]) -> None:
        """Run the flash (0.5 s) to the state that waits for the capture's depth, 0.3 s after its start."""
        self.sequencer.settings.flash_in = 0.5
        self.sequencer.set_sequence(SequenceType.FLASH)
        self.sequencer.update()
        clock[0] += 0.3
        for _ in range(3):      # past the half: CAPTURE, then ASK
            self.sequencer.update()
        self.assertEqual(self.sequencer.get_sequence(), SequenceType.ASK)

    def test_flash_waits_for_the_depth(self):
        clock = [100.0]
        with mock.patch('modules.render.Sequencer.time.time', lambda: clock[0]):
            self.flash_until_ask(clock)
            clock[0] += 10.0
            self.sequencer.update()
            self.assertEqual(self.sequencer.get_sequence(), SequenceType.ASK)

    def test_flash_out_starts_after_the_hold(self):
        clock = [100.0]
        self.sequencer.settings.flash_hold = 1.0
        with mock.patch('modules.render.Sequencer.time.time', lambda: clock[0]):
            self.flash_until_ask(clock)
            self.sequencer.start_reveal()       # the depth is in before the flash is full
            clock[0] = 101.4                    # flash full at 100.5, held until 101.5
            self.sequencer.update()
            self.assertEqual(self.sequencer.get_sequence(), SequenceType.ASK)
            clock[0] = 101.6
            self.assertEqual(self.sequencer.update().sequence_type, SequenceType.ASK)
            self.assertEqual(self.sequencer.get_sequence(), SequenceType.CONTROL_0)
            message = self.sequencer.update()
            self.assertEqual(message.sequence_type, SequenceType.CONTROL_0)
            self.assertTrue(message.initial)

    def test_a_depth_from_before_the_flash_does_not_start_the_flash_out(self):
        clock = [100.0]
        self.sequencer.settings.flash_hold = 0.0
        self.sequencer.start_reveal()
        with mock.patch('modules.render.Sequencer.time.time', lambda: clock[0]):
            self.flash_until_ask(clock)
            clock[0] += 5.0
            self.sequencer.update()
            self.assertEqual(self.sequencer.get_sequence(), SequenceType.ASK)
