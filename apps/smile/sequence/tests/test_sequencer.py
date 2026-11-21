"""The sequencer's message callbacks: a `utils.Broadcast`, so one that fails leaves the next one called, and they run
outside the sequencer's lock (a callback that calls back into the sequencer does not wait)."""

import unittest

from apps.smile.sequence import SequenceMessage, Sequencer, SequencerSettings


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
            self.sequencer.update()
        self.assertEqual(len(heard), 1)

    def test_a_callback_may_remove_itself(self) -> None:
        # the callbacks are called outside the list's lock: a callback that changes the list does not wait for it
        heard: list[SequenceMessage] = []

        def once(message: SequenceMessage) -> None:
            heard.append(message)
            self.sequencer.remove_message_callback(once)
        self.sequencer.add_message_callback(once)
        self.sequencer.update()
        self.sequencer.update()
        self.assertEqual(len(heard), 1)

    def test_cleared_callbacks_hear_nothing(self) -> None:
        heard: list[SequenceMessage] = []
        self.sequencer.add_message_callback(heard.append)
        self.sequencer.clear_message_callbacks()
        self.sequencer.update()
        self.assertEqual(heard, [])


if __name__ == '__main__':
    unittest.main()
