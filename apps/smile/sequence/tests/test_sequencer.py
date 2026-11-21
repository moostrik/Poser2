"""The sequencer: its state changes, stepped through time without sleeping (each update gets its tick, as the render
gives it), and its message callbacks, a `utils.Broadcast`: one that fails leaves the next one called, and they run
outside the sequencer's lock."""

import unittest

from apps.smile.sequence import SequenceMessage, Sequencer, SequencerSettings, SequenceType


class TestSequencer(unittest.TestCase):
    def setUp(self):
        self.sequencer = Sequencer(SequencerSettings())
        self.now: float = 100.0

    def tearDown(self):
        self.sequencer.close()

    def update(self, step: float = 0.0) -> SequenceMessage:
        """One render frame, `step` s after the last."""
        self.now += step
        return self.sequencer.update(self.now)

    def run_camera(self, seconds: float, face: bool | None, step: float = 0.1) -> SequenceMessage:
        """Step the sequencer through `seconds` of the camera phase; `face` is what the detector
        reports every step (None: it reports nothing). Returns the last message."""
        message = self.update()
        for _ in range(round(seconds / step)):
            self.now += step
            if face is not None:
                self.sequencer.set_face(face, self.now)
            message = self.update()
        return message

    def charging(self, charge: float = 2.0, drain: float = 4.0) -> None:
        self.sequencer.settings.charge = charge
        self.sequencer.settings.drain = drain

    def test_the_message_carries_the_tick_it_was_made_at(self):
        self.assertEqual(self.update(0.25).timestamp, 100.25)

    def test_a_qualifying_face_charges_to_the_flash(self):
        self.charging()
        message = self.run_camera(1.0, True)
        self.assertEqual(message.sequence_type, SequenceType.CAMERA_DETECT)
        self.assertAlmostEqual(message.progress, 0.5, places=5)
        self.assertAlmostEqual(message.value_1, 0.5, places=5)
        self.run_camera(1.1, True)
        self.assertEqual(self.sequencer.get_sequence(), SequenceType.FLASH)

    def test_the_charge_drains_slower_than_it_rises_and_ends_in_camera_show(self):
        self.charging()
        self.run_camera(1.0, True)                       # half full
        message = self.run_camera(1.0, False)            # a quarter lost
        self.assertEqual(message.sequence_type, SequenceType.CAMERA_DETECT)
        self.assertAlmostEqual(message.progress, 0.25, places=5)
        message = self.run_camera(1.5, False)
        self.assertEqual(message.sequence_type, SequenceType.CAMERA_SHOW)
        self.assertEqual(message.value_1, 0.0)
        self.assertEqual(self.sequencer.get_sequence(), SequenceType.CAMERA_SHOW)

    def test_a_charge_that_restarts_is_initial_again(self):
        self.charging()
        self.run_camera(0.2, True)
        self.run_camera(1.0, False)                      # drained
        self.sequencer.set_face(True, self.now)
        message = self.update(0.1)
        self.assertEqual(message.sequence_type, SequenceType.CAMERA_DETECT)
        self.assertTrue(message.initial)

    def test_a_detector_that_stopped_reporting_counts_as_no_face(self):
        self.charging()
        self.run_camera(1.0, True)
        message = self.run_camera(2.0, None)             # the last report goes stale
        self.assertLess(message.progress, 0.5)
        self.assertEqual(self.sequencer.get_sequence(), SequenceType.CAMERA_DETECT)

    def test_pause_ignores_a_face_but_not_the_start(self):
        self.charging()
        self.sequencer.settings.pause = True
        message = self.run_camera(1.0, True)
        self.assertEqual(message.sequence_type, SequenceType.CAMERA_SHOW)
        self.sequencer.start_sequence()
        self.run_camera(2.1, None)                       # the start needs no face
        self.assertEqual(self.sequencer.get_sequence(), SequenceType.FLASH)

    def test_starts_only_in_the_camera_phase(self):
        self.charging()
        self.sequencer.set_sequence(SequenceType.DIFFUSION)
        self.sequencer.start_sequence()
        self.sequencer.set_sequence(SequenceType.CAMERA_SHOW)
        message = self.run_camera(1.0, None)
        self.assertEqual(message.sequence_type, SequenceType.CAMERA_SHOW)

    def test_blocked_charges_nothing(self):
        self.charging()
        self.sequencer.blocked = True
        self.sequencer.start_sequence()
        message = self.run_camera(1.0, True)
        self.assertEqual(message.sequence_type, SequenceType.CAMERA_SHOW)
        # a sequence that is running goes on
        self.sequencer.set_sequence(SequenceType.DIFFUSION)
        self.assertEqual(self.sequencer.get_sequence(), SequenceType.DIFFUSION)
        self.sequencer.set_sequence(SequenceType.CAMERA_SHOW)
        self.sequencer.blocked = False
        message = self.run_camera(1.0, True)
        self.assertEqual(message.sequence_type, SequenceType.CAMERA_DETECT)

    def test_a_start_that_was_blocked_does_not_fire_later(self):
        self.charging()
        self.sequencer.start_sequence()
        self.run_camera(1.0, None)                       # half full
        self.sequencer.blocked = True
        self.run_camera(3.0, None)                       # drained
        self.sequencer.blocked = False
        message = self.run_camera(3.0, None)
        self.assertEqual(message.sequence_type, SequenceType.CAMERA_SHOW)

    def test_progress_stays_full_from_the_timeout_into_the_reset(self):
        self.sequencer.settings.timeout = 1.0
        self.sequencer.set_sequence(SequenceType.TIMEOUT)
        self.update()
        self.now += 1.1
        for _ in range(2):      # the lights and the music follow the progress of both RESET messages
            message = self.update()
            self.assertEqual(message.sequence_type, SequenceType.RESET)
            self.assertEqual(message.progress, 1.0)
        self.assertEqual(self.sequencer.get_sequence(), SequenceType.BLOOM)

    def test_the_first_fade_message_keeps_the_bloom(self):
        settings = self.sequencer.settings
        settings.bloom = 0.4
        settings.fade = 1.1
        self.sequencer.set_sequence(SequenceType.BLOOM)
        self.update()
        bloom = self.update(0.39)
        self.assertEqual(bloom.sequence_type, SequenceType.BLOOM)
        first = self.update(0.02)                       # the bloom's last message is a FADE
        self.assertEqual(first.sequence_type, SequenceType.FADE)
        self.assertEqual(first.value_1, 0.0)            # nothing faded yet
        self.assertGreater(first.value_2, bloom.value_1)    # the bloom goes on
        second = self.update()
        self.assertEqual(second.sequence_type, SequenceType.FADE)
        self.assertEqual(second.value_2, first.value_2)

    def test_update_reports_state_and_calls_back(self):
        messages = []
        self.sequencer.add_message_callback(messages.append)
        message = self.update()
        self.assertEqual(message.sequence_type, SequenceType.CAMERA_SHOW)
        self.assertTrue(message.initial)
        self.assertEqual(messages, [message])
        self.assertIs(self.sequencer.message, message)
        self.assertEqual(self.sequencer.settings.state, 'CAMERA_SHOW')

    def flash_until_ask(self) -> None:
        """Run the flash (0.5 s) to the state that waits for the capture's depth, 0.3 s after its start."""
        self.sequencer.settings.flash_in = 0.5
        self.sequencer.set_sequence(SequenceType.FLASH)
        self.update()
        self.now += 0.3
        for _ in range(3):      # past the half: CAPTURE, then ASK
            self.update()
        self.assertEqual(self.sequencer.get_sequence(), SequenceType.ASK)

    def test_flash_waits_for_the_depth(self):
        self.flash_until_ask()
        self.update(10.0)
        self.assertEqual(self.sequencer.get_sequence(), SequenceType.ASK)

    def test_flash_out_starts_after_the_hold(self):
        self.sequencer.settings.flash_hold = 1.0
        self.flash_until_ask()
        self.sequencer.start_reveal()       # the depth is in before the flash is full
        self.now = 101.4                    # flash full at 100.5, held until 101.5
        self.update()
        self.assertEqual(self.sequencer.get_sequence(), SequenceType.ASK)
        self.now = 101.6
        self.assertEqual(self.update().sequence_type, SequenceType.ASK)
        self.assertEqual(self.sequencer.get_sequence(), SequenceType.CONTROL_0)
        message = self.update()
        self.assertEqual(message.sequence_type, SequenceType.CONTROL_0)
        self.assertTrue(message.initial)

    def test_finish_with_the_depth_in_lets_the_flash_out_play(self):
        settings = self.sequencer.settings
        settings.flash_hold = 0.0
        settings.flash_out_auto = False
        settings.flash_out = 2.0
        self.flash_until_ask()
        self.sequencer.start_reveal()
        self.sequencer.finish()             # nothing generated: done before the flash is full
        self.assertEqual(self.sequencer.get_sequence(), SequenceType.ASK)
        self.update(0.3)
        self.assertEqual(self.sequencer.get_sequence(), SequenceType.CONTROL_0)
        message = self.update()
        self.assertEqual(message.sequence_type, SequenceType.CONTROL_0)
        self.assertTrue(message.initial)
        self.update(1.0)
        self.assertEqual(self.sequencer.get_sequence(), SequenceType.CONTROL_0)
        self.update(1.1)                    # the flash out is done
        self.assertEqual(self.sequencer.get_sequence(), SequenceType.TIMEOUT)

    def test_finish_without_the_depth_frees_the_flash(self):
        self.flash_until_ask()
        self.sequencer.finish()             # the generator failed before the depth
        self.assertEqual(self.sequencer.get_sequence(), SequenceType.TIMEOUT)

    def test_finish_after_the_flash_is_at_once(self):
        for state in (SequenceType.DIFFUSION, SequenceType.DIFF_LOOP):
            self.sequencer.start_reveal()
            self.sequencer.set_sequence(state)
            self.sequencer.finish()
            self.assertEqual(self.sequencer.get_sequence(), SequenceType.TIMEOUT)

    def test_a_finish_of_the_previous_visitor_does_not_end_the_next_flash_out(self):
        settings = self.sequencer.settings
        settings.flash_hold = 0.0
        settings.flash_out_auto = False
        settings.flash_out = 2.0
        self.flash_until_ask()
        self.sequencer.start_reveal()
        self.sequencer.finish()
        self.flash_until_ask()         # the next visitor
        self.sequencer.start_reveal()
        self.update(0.3)
        self.update()
        self.update(10.0)
        self.assertEqual(self.sequencer.get_sequence(), SequenceType.CONTROL_0)

    def test_flash_out_waits_for_the_generator(self):
        self.sequencer.settings.flash_out_auto = False
        self.sequencer.set_sequence(SequenceType.CONTROL_0)
        self.update()
        self.update(10.0)
        self.assertEqual(self.sequencer.get_sequence(), SequenceType.CONTROL_0)

    def test_a_depth_from_before_the_flash_does_not_start_the_flash_out(self):
        self.sequencer.settings.flash_hold = 0.0
        self.sequencer.start_reveal()
        self.flash_until_ask()
        self.update(5.0)
        self.assertEqual(self.sequencer.get_sequence(), SequenceType.ASK)


class TestSequencerCallbacks(unittest.TestCase):
    def setUp(self) -> None:
        self.sequencer = Sequencer(SequencerSettings())

    def test_a_failing_callback_leaves_the_next_one_called(self) -> None:
        def fail(_message: SequenceMessage) -> None:
            raise RuntimeError('a listener failed')
        heard: list[SequenceMessage] = []
        self.sequencer.add_message_callback(fail)
        self.sequencer.add_message_callback(heard.append)
        with self.assertLogs('modules.utils.broadcast', level='ERROR'):
            self.sequencer.update(0.0)
        self.assertEqual(len(heard), 1)

    def test_a_callback_may_remove_itself(self) -> None:
        # the callbacks are called outside the list's lock: a callback that changes the list does not wait for it
        heard: list[SequenceMessage] = []

        def once(message: SequenceMessage) -> None:
            heard.append(message)
            self.sequencer.remove_message_callback(once)
        self.sequencer.add_message_callback(once)
        self.sequencer.update(0.0)
        self.sequencer.update(0.1)
        self.assertEqual(len(heard), 1)

    def test_cleared_callbacks_hear_nothing(self) -> None:
        heard: list[SequenceMessage] = []
        self.sequencer.add_message_callback(heard.append)
        self.sequencer.clear_message_callbacks()
        self.sequencer.update(0.0)
        self.assertEqual(heard, [])


if __name__ == '__main__':
    unittest.main()
