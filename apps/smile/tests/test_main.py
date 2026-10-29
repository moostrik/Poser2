"""Tests for the app's limit on a visitor the generator does not end (no camera, no GL, no ComfyUI:
the method runs on a stand-in for the app)."""

import unittest
from types import SimpleNamespace
from unittest import mock

from apps.smile.comfy.settings import GeneratorSettings
from apps.smile.main import VISITOR_MARGIN, SmileMain
from apps.smile.render.Sequencer import SequenceType


class TestVisitorLimit(unittest.TestCase):
    def setUp(self):
        generator = GeneratorSettings()
        generator.connection.job_timeout = 45.0
        self.app = SimpleNamespace(settings=SimpleNamespace(generator=generator), generator=mock.Mock(),
                                   sequencer=mock.Mock(), _visitor_start=100.0)
        self.limit: float = 100.0 + 45.0 + VISITOR_MARGIN

    def step(self, now: float, seq: SequenceType) -> None:
        with mock.patch('apps.smile.main.time.perf_counter', lambda: now):
            SmileMain._limit_visitor(self.app, seq)

    def test_a_visitor_within_its_time_is_left_alone(self):
        self.step(self.limit - 1.0, SequenceType.DIFFUSION)
        self.app.sequencer.finish.assert_not_called()
        self.assertEqual(self.app._visitor_start, 100.0)

    def test_a_visitor_the_generator_did_not_end_is_ended_once(self):
        for seq in (SequenceType.ASK, SequenceType.ASK):
            self.step(self.limit + 1.0, seq)
        self.app.generator.interrupt.assert_called_once()
        self.app.sequencer.finish.assert_called_once()

    def test_a_visitor_that_ended_has_no_limit(self):
        self.step(110.0, SequenceType.TIMEOUT)
        self.assertIsNone(self.app._visitor_start)
        self.step(self.limit + 1.0, SequenceType.CAMERA_SHOW)
        self.app.sequencer.finish.assert_not_called()

    def test_the_grin_loop_has_no_limit_and_gets_the_full_time_after_it(self):
        self.app.settings.generator.grin.loop = True
        self.step(self.limit + 100.0, SequenceType.DIFF_LOOP)
        self.app.sequencer.finish.assert_not_called()
        self.app.settings.generator.grin.loop = False
        self.step(self.limit + 101.0, SequenceType.DIFF_LOOP)     # the pass that was running may finish
        self.app.sequencer.finish.assert_not_called()


if __name__ == '__main__':
    unittest.main()
