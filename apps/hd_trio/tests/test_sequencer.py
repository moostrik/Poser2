"""The show's sequencer: the panel's requests carried out at the tick, from any thread, in order."""

import threading
import unittest

from modules.session import SequencerState

from ..sequencer import Sequencer, SequencerSettings


def settings(durations: list[float]) -> SequencerSettings:
    s = SequencerSettings()
    s.stages = list(range(len(durations)))
    s.durations = durations
    s.initialize()
    return s


class SequencerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = settings([1.0, 1.0, 1.0])
        self.sequencer = Sequencer(self.settings)
        self.states: list[SequencerState] = []
        self.sequencer.add_state_callback(self.states.append)

    def tearDown(self) -> None:
        self.sequencer.stop()

    def test_a_start_from_another_thread_begins_at_the_next_tick(self) -> None:
        thread = threading.Thread(target=SequencerSettings.start.fire, args=(self.settings,))
        thread.start()
        thread.join()
        self.assertFalse(self.settings.running)                  # nothing changed on the panel's thread
        self.sequencer.update(10.0)
        self.assertTrue(self.settings.running)
        self.assertEqual(self.settings.stage, 0)
        self.sequencer.update(10.5)
        self.assertAlmostEqual(self.settings.stage_progress, 0.5)  # timed from the tick it started on

    def test_the_stages_follow_in_order_while_requests_come_from_another_thread(self) -> None:
        SequencerSettings.start.fire(self.settings)
        now = 0.0
        stop = threading.Event()

        def skip_now_and_then() -> None:
            while not stop.is_set():
                SequencerSettings.skip.fire(self.settings)
                stop.wait(0.001)
        thread = threading.Thread(target=skip_now_and_then)
        thread.start()
        for _ in range(200):
            self.sequencer.update(now)
            now += 0.05
        stop.set()
        thread.join()
        active = [state.stage for state in self.states if state.active]
        self.assertTrue(all(b in (a, a + 1) for a, b in zip(active, active[1:])))      # never back, never a stage left out

    def test_the_end_of_the_show_stops_it(self) -> None:
        SequencerSettings.start.fire(self.settings)
        now = 0.0
        for _ in range(100):
            self.sequencer.update(now)
            now += 0.1
        self.assertFalse(self.settings.running)

    def test_stop_unbinds_the_settings(self) -> None:
        self.sequencer.stop()
        SequencerSettings.start.fire(self.settings)
        self.sequencer.update(0.0)
        self.assertFalse(self.settings.running)
        self.sequencer = Sequencer(self.settings)        # for tearDown


if __name__ == '__main__':
    unittest.main()
