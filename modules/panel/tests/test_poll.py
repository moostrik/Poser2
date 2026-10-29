"""The poll tick: every control is polled, a failing control is logged once and skipped, the others go on."""

import logging
import unittest

from ..poll import tick


class FakeControl:
    def __init__(self, name: str, fail: bool = False) -> None:
        self.settings = self
        self.name = name
        self.fail = fail
        self.polled = 0

    def poll(self) -> None:
        if self.fail:
            raise RuntimeError("client gone")
        self.polled += 1


class TestTick(unittest.TestCase):
    def test_every_control_is_polled(self):
        a, b = FakeControl("a"), FakeControl("b")
        failed: set[int] = set()
        tick([a, b], failed)
        tick([a, b], failed)
        self.assertEqual((a.polled, b.polled), (2, 2))
        self.assertEqual(failed, set())

    def test_failing_control_is_logged_once_and_the_others_go_on(self):
        a, b = FakeControl("a", fail=True), FakeControl("b")
        failed: set[int] = set()
        with self.assertLogs("modules.panel.poll", level=logging.ERROR) as logs:
            tick([a, b], failed)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("FakeControl.a", logs.output[0])
        self.assertEqual(b.polled, 1)
        self.assertEqual(failed, {0})

        with self.assertNoLogs("modules.panel.poll", level=logging.ERROR):
            tick([a, b], failed)
        self.assertEqual(b.polled, 2)

    def test_recovered_control_logs_again_on_its_next_failure(self):
        a = FakeControl("a", fail=True)
        failed: set[int] = set()
        with self.assertLogs("modules.panel.poll", level=logging.ERROR):
            tick([a], failed)
        a.fail = False
        with self.assertNoLogs("modules.panel.poll", level=logging.ERROR):
            tick([a], failed)
        self.assertEqual(failed, set())
        a.fail = True
        with self.assertLogs("modules.panel.poll", level=logging.ERROR):
            tick([a], failed)
        self.assertEqual(failed, {0})


if __name__ == "__main__":
    unittest.main()
