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

    def run_camera(self, clock: list[float], seconds: float, face: bool | None, step: float = 0.1):
        """Step the sequencer through `seconds` of the camera phase; `face` is what the detector
        reports every step (None: it reports nothing). Returns the last message."""
        message = self.sequencer.update()
        for _ in range(round(seconds / step)):
            clock[0] += step
            if face is not None:
                self.sequencer.set_face(face, clock[0])
            message = self.sequencer.update()
        return message

    def charging(self, charge: float = 2.0, drain: float = 4.0):
        self.sequencer.settings.charge = charge
        self.sequencer.settings.drain = drain
        clock = [100.0]
        return clock, mock.patch('modules.render.Sequencer.time.time', lambda: clock[0])

    def test_a_qualifying_face_charges_to_the_flash(self):
        clock, patched = self.charging()
        with patched:
            message = self.run_camera(clock, 1.0, True)
            self.assertEqual(message.sequence_type, SequenceType.CAMERA_DETECT)
            self.assertAlmostEqual(message.progress, 0.5, places=5)
            self.assertAlmostEqual(message.value_1, 0.5, places=5)
            self.run_camera(clock, 1.1, True)
            self.assertEqual(self.sequencer.get_sequence(), SequenceType.FLASH)

    def test_the_charge_drains_slower_than_it_rises_and_ends_in_camera_show(self):
        clock, patched = self.charging()
        with patched:
            self.run_camera(clock, 1.0, True)                       # half full
            message = self.run_camera(clock, 1.0, False)            # a quarter lost
            self.assertEqual(message.sequence_type, SequenceType.CAMERA_DETECT)
            self.assertAlmostEqual(message.progress, 0.25, places=5)
            message = self.run_camera(clock, 1.5, False)
            self.assertEqual(message.sequence_type, SequenceType.CAMERA_SHOW)
            self.assertEqual(message.value_1, 0.0)
            self.assertEqual(self.sequencer.get_sequence(), SequenceType.CAMERA_SHOW)

    def test_a_charge_that_restarts_is_initial_again(self):
        clock, patched = self.charging()
        with patched:
            self.run_camera(clock, 0.2, True)
            self.run_camera(clock, 1.0, False)                      # drained
            self.sequencer.set_face(True, clock[0])
            clock[0] += 0.1
            message = self.sequencer.update()
            self.assertEqual(message.sequence_type, SequenceType.CAMERA_DETECT)
            self.assertTrue(message.initial)

    def test_a_detector_that_stopped_reporting_counts_as_no_face(self):
        clock, patched = self.charging()
        with patched:
            self.run_camera(clock, 1.0, True)
            message = self.run_camera(clock, 2.0, None)             # the last report goes stale
            self.assertLess(message.progress, 0.5)
            self.assertEqual(self.sequencer.get_sequence(), SequenceType.CAMERA_DETECT)

    def test_pause_ignores_a_face_but_not_the_start(self):
        clock, patched = self.charging()
        self.sequencer.settings.pause = True
        with patched:
            message = self.run_camera(clock, 1.0, True)
            self.assertEqual(message.sequence_type, SequenceType.CAMERA_SHOW)
            self.sequencer.start_sequence()
            self.run_camera(clock, 2.1, None)                       # the start needs no face
            self.assertEqual(self.sequencer.get_sequence(), SequenceType.FLASH)

    def test_starts_only_in_the_camera_phase(self):
        clock, patched = self.charging()
        with patched:
            self.sequencer.set_sequence(SequenceType.DIFFUSION)
            self.sequencer.start_sequence()
            self.sequencer.set_sequence(SequenceType.CAMERA_SHOW)
            message = self.run_camera(clock, 1.0, None)
            self.assertEqual(message.sequence_type, SequenceType.CAMERA_SHOW)

    def test_blocked_charges_nothing(self):
        clock, patched = self.charging()
        self.sequencer.blocked = True
        with patched:
            self.sequencer.start_sequence()
            message = self.run_camera(clock, 1.0, True)
            self.assertEqual(message.sequence_type, SequenceType.CAMERA_SHOW)
            # a sequence that is running goes on
            self.sequencer.set_sequence(SequenceType.DIFFUSION)
            self.assertEqual(self.sequencer.get_sequence(), SequenceType.DIFFUSION)
            self.sequencer.set_sequence(SequenceType.CAMERA_SHOW)
            self.sequencer.blocked = False
            message = self.run_camera(clock, 1.0, True)
            self.assertEqual(message.sequence_type, SequenceType.CAMERA_DETECT)

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
