"""The poll tick: changed values reach their controls, a failing control is logged once and skipped."""

import logging
import unittest

from ..poll import tick


class Source:
    def __init__(self) -> None:
        self.a = 1
        self.b = "x"


class TestTick(unittest.TestCase):
    def setUp(self):
        self.source = Source()
        self.shown: list[tuple[str, object]] = []
        self.polls = [
            (self.source, "a", [self.source.a], lambda v: self.shown.append(("a", v))),
            (self.source, "b", [self.source.b], lambda v: self.shown.append(("b", v))),
        ]
        self.failed: set[int] = set()

    def test_unchanged_values_are_not_shown(self):
        tick(self.polls, self.failed)
        self.assertEqual(self.shown, [])

    def test_changed_value_is_shown_once(self):
        self.source.a = 2
        tick(self.polls, self.failed)
        tick(self.polls, self.failed)
        self.assertEqual(self.shown, [("a", 2)])
        self.assertEqual(self.polls[0][2], [2])

    def test_a_committed_value_is_not_pushed_back(self):
        self.source.a = 3
        self.polls[0][2][0] = 3     # the control recorded what it wrote
        tick(self.polls, self.failed)
        self.assertEqual(self.shown, [])

    def test_failing_control_is_logged_once_and_the_others_go_on(self):
        def broken(_v):
            raise RuntimeError("client gone")
        self.polls[0] = (self.source, "a", [self.source.a], broken)
        self.source.a = 2
        self.source.b = "y"
        with self.assertLogs("modules.panel.poll", level=logging.ERROR) as logs:
            tick(self.polls, self.failed)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("Source.a", logs.output[0])
        self.assertEqual(self.shown, [("b", "y")])
        self.assertEqual(self.failed, {0})

        self.source.a = 3
        with self.assertNoLogs("modules.panel.poll", level=logging.ERROR):
            tick(self.polls, self.failed)
        self.assertEqual(self.failed, {0})

    def test_recovered_control_logs_again_on_its_next_failure(self):
        calls = {"n": 0}

        def flaky(_v):
            calls["n"] += 1
            if calls["n"] != 2:
                raise RuntimeError("client gone")
        self.polls[0] = (self.source, "a", [self.source.a], flaky)
        for value in (2, 3, 4):
            self.source.a = value
            with self.assertLogs("modules.panel.poll", level=logging.ERROR) if value != 3 \
                    else self.assertNoLogs("modules.panel.poll", level=logging.ERROR):
                tick(self.polls, self.failed)
        self.assertEqual(self.failed, {0})


if __name__ == "__main__":
    unittest.main()
