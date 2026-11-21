"""Tests for the render's sequence to stage table (no GL needed); the sequencer's own are in sequence/tests."""

import unittest
from unittest import mock

from apps.smile.sequence import SequenceMessage, SequenceType
from apps.smile.render.stages import STAGES, CameraStage, BloomStage, DiffusionStage, FlashStage, RevealStage


class TestStages(unittest.TestCase):
    def test_every_sequence_type_has_a_stage(self):
        for seq in SequenceType:
            self.assertIn(seq, STAGES, seq.name)

    def test_cycle_stages(self):
        self.assertIs(STAGES[SequenceType.CAMERA_SHOW], CameraStage)
        self.assertIs(STAGES[SequenceType.CAPTURE], FlashStage)
        self.assertIs(STAGES[SequenceType.CONTROL_0], RevealStage)
        self.assertIs(STAGES[SequenceType.DIFF_LOOP], DiffusionStage)
        self.assertIs(STAGES[SequenceType.FADE], BloomStage)

    def test_the_capture_seeds_the_diffusion_layer(self):
        # at the capture, not at the flash out: a generation that fails before its depth skips that
        inputs, layers = mock.Mock(), mock.Mock()
        stage = FlashStage(inputs, layers, mock.Mock())
        stage.update(SequenceMessage(SequenceType.FLASH, 0.2))
        stage.update(SequenceMessage(SequenceType.ASK, 0.8))
        layers.diffusion.seed.assert_not_called()
        stage.update(SequenceMessage(SequenceType.CAPTURE, 0.5))
        layers.diffusion.seed.assert_called_once_with(inputs.capture)


if __name__ == '__main__':
    unittest.main()
