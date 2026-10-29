"""Tests for the render's sequence to stage table and the sequencer's state changes (no GL needed)."""

import unittest
from unittest import mock

from apps.smile.render.Sequencer import SequenceMessage, Sequencer, SequenceType
from apps.smile.render.settings import SequencerSettings
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
        return clock, mock.patch('apps.smile.render.Sequencer.time.perf_counter', lambda: clock[0])

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

    def test_a_start_that_was_blocked_does_not_fire_later(self):
        clock, patched = self.charging()
        with patched:
            self.sequencer.start_sequence()
            self.run_camera(clock, 1.0, None)                       # half full
            self.sequencer.blocked = True
            self.run_camera(clock, 3.0, None)                       # drained
            self.sequencer.blocked = False
            message = self.run_camera(clock, 3.0, None)
            self.assertEqual(message.sequence_type, SequenceType.CAMERA_SHOW)

    def test_progress_stays_full_from_the_timeout_into_the_reset(self):
        clock = [100.0]
        self.sequencer.settings.timeout = 1.0
        with mock.patch('apps.smile.render.Sequencer.time.perf_counter', lambda: clock[0]):
            self.sequencer.set_sequence(SequenceType.TIMEOUT)
            self.sequencer.update()
            clock[0] += 1.1
            for _ in range(2):      # the lights and the music follow the progress of both RESET messages
                message = self.sequencer.update()
                self.assertEqual(message.sequence_type, SequenceType.RESET)
                self.assertEqual(message.progress, 1.0)
            self.assertEqual(self.sequencer.get_sequence(), SequenceType.BLOOM)

    def test_the_first_fade_message_keeps_the_bloom(self):
        clock = [100.0]
        settings = self.sequencer.settings
        settings.bloom = 0.4
        settings.fade = 1.1
        with mock.patch('apps.smile.render.Sequencer.time.perf_counter', lambda: clock[0]):
            self.sequencer.set_sequence(SequenceType.BLOOM)
            self.sequencer.update()
            clock[0] += 0.39
            bloom = self.sequencer.update()
            self.assertEqual(bloom.sequence_type, SequenceType.BLOOM)
            clock[0] += 0.02
            first = self.sequencer.update()                         # the bloom's last message is a FADE
            self.assertEqual(first.sequence_type, SequenceType.FADE)
            self.assertEqual(first.value_1, 0.0)                    # nothing faded yet
            self.assertGreater(first.value_2, bloom.value_1)        # the bloom goes on
            second = self.sequencer.update()
            self.assertEqual(second.sequence_type, SequenceType.FADE)
            self.assertEqual(second.value_2, first.value_2)

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
        with mock.patch('apps.smile.render.Sequencer.time.perf_counter', lambda: clock[0]):
            self.flash_until_ask(clock)
            clock[0] += 10.0
            self.sequencer.update()
            self.assertEqual(self.sequencer.get_sequence(), SequenceType.ASK)

    def test_flash_out_starts_after_the_hold(self):
        clock = [100.0]
        self.sequencer.settings.flash_hold = 1.0
        with mock.patch('apps.smile.render.Sequencer.time.perf_counter', lambda: clock[0]):
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

    def test_finish_with_the_depth_in_lets_the_flash_out_play(self):
        clock = [100.0]
        settings = self.sequencer.settings
        settings.flash_hold = 0.0
        settings.flash_out_auto = False
        settings.flash_out = 2.0
        with mock.patch('apps.smile.render.Sequencer.time.perf_counter', lambda: clock[0]):
            self.flash_until_ask(clock)
            self.sequencer.start_reveal()
            self.sequencer.finish()             # nothing generated: done before the flash is full
            self.assertEqual(self.sequencer.get_sequence(), SequenceType.ASK)
            clock[0] += 0.3
            self.sequencer.update()
            self.assertEqual(self.sequencer.get_sequence(), SequenceType.CONTROL_0)
            message = self.sequencer.update()
            self.assertEqual(message.sequence_type, SequenceType.CONTROL_0)
            self.assertTrue(message.initial)
            clock[0] += 1.0
            self.sequencer.update()
            self.assertEqual(self.sequencer.get_sequence(), SequenceType.CONTROL_0)
            clock[0] += 1.1                     # the flash out is done
            self.sequencer.update()
            self.assertEqual(self.sequencer.get_sequence(), SequenceType.TIMEOUT)

    def test_finish_without_the_depth_frees_the_flash(self):
        clock = [100.0]
        with mock.patch('apps.smile.render.Sequencer.time.perf_counter', lambda: clock[0]):
            self.flash_until_ask(clock)
            self.sequencer.finish()             # the generator failed before the depth
            self.assertEqual(self.sequencer.get_sequence(), SequenceType.TIMEOUT)

    def test_finish_after_the_flash_is_at_once(self):
        for state in (SequenceType.DIFFUSION, SequenceType.DIFF_LOOP):
            self.sequencer.start_reveal()
            self.sequencer.set_sequence(state)
            self.sequencer.finish()
            self.assertEqual(self.sequencer.get_sequence(), SequenceType.TIMEOUT)

    def test_a_finish_of_the_previous_visitor_does_not_end_the_next_flash_out(self):
        clock = [100.0]
        settings = self.sequencer.settings
        settings.flash_hold = 0.0
        settings.flash_out_auto = False
        settings.flash_out = 2.0
        with mock.patch('apps.smile.render.Sequencer.time.perf_counter', lambda: clock[0]):
            self.flash_until_ask(clock)
            self.sequencer.start_reveal()
            self.sequencer.finish()
            self.flash_until_ask(clock)         # the next visitor
            self.sequencer.start_reveal()
            clock[0] += 0.3
            self.sequencer.update()
            self.sequencer.update()
            clock[0] += 10.0
            self.sequencer.update()
            self.assertEqual(self.sequencer.get_sequence(), SequenceType.CONTROL_0)

    def test_flash_out_waits_for_the_generator(self):
        clock = [100.0]
        self.sequencer.settings.flash_out_auto = False
        with mock.patch('apps.smile.render.Sequencer.time.perf_counter', lambda: clock[0]):
            self.sequencer.set_sequence(SequenceType.CONTROL_0)
            self.sequencer.update()
            clock[0] += 10.0
            self.sequencer.update()
            self.assertEqual(self.sequencer.get_sequence(), SequenceType.CONTROL_0)

    def test_a_depth_from_before_the_flash_does_not_start_the_flash_out(self):
        clock = [100.0]
        self.sequencer.settings.flash_hold = 0.0
        self.sequencer.start_reveal()
        with mock.patch('apps.smile.render.Sequencer.time.perf_counter', lambda: clock[0]):
            self.flash_until_ask(clock)
            clock[0] += 5.0
            self.sequencer.update()
            self.assertEqual(self.sequencer.get_sequence(), SequenceType.ASK)
