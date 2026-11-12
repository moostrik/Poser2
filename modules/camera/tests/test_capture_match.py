"""`CaptureStream` and `MissedParts`: a part's values found by capture time, and the parts no capture took, counted."""

import unittest

from modules.camera import CAPTURE_TOLERANCE, CaptureStream, MissedParts


class CaptureStreamTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stream: CaptureStream[str] = CaptureStream(4)
        for t, value in ((1.0, 'a'), (2.0, 'b'), (3.0, 'c')):
            self.stream.add(t, value)

    def test_a_value_is_found_within_the_tolerance(self) -> None:
        self.assertEqual(self.stream.at(2.0 + CAPTURE_TOLERANCE / 2), 'b')
        self.assertIsNone(self.stream.at(2.5))

    def test_before_is_the_newest_captured_earlier(self) -> None:
        self.assertEqual(self.stream.before(2.5), 'b')
        self.assertIsNone(self.stream.before(1.0))

    def test_it_keeps_up_while_its_newest_is_as_new_as_the_last_capture(self) -> None:
        self.assertTrue(self.stream.keeps_up(3.0))
        self.assertFalse(self.stream.keeps_up(3.1))
        self.assertFalse(CaptureStream(4).keeps_up(0.0))

    def test_taking_through_a_capture_counts_what_came_before_it(self) -> None:
        self.assertEqual(self.stream.take_through(2.0), 1)      # 'a' was never taken, 'b' is the capture's
        self.assertEqual([value for _, value in self.stream], ['c'])

    def test_it_keeps_only_the_newest(self) -> None:
        self.stream.add(4.0, 'd')
        self.stream.add(5.0, 'e')
        self.assertEqual([value for _, value in self.stream], ['b', 'c', 'd', 'e'])


class MissedPartsTest(unittest.TestCase):
    def test_the_start_is_counted_apart(self) -> None:
        missed = MissedParts('test')
        missed.count(0, 'depth', 3)
        missed.started()
        with self.assertLogs('modules.camera.capture_match', 'WARNING'):
            missed.count(0, 'depth')
        self.assertEqual((missed.at_start('depth'), missed.total('depth')), (3, 1))

    def test_counts_per_part_and_a_reset(self) -> None:
        missed = MissedParts('test')
        missed.started()
        with self.assertLogs('modules.camera.capture_match', 'WARNING'):
            missed.count(0, 'depth')
            missed.count(1, 'depth', 2)
            missed.count(0, 'tracklets')
        self.assertEqual((missed.total('depth'), missed.total('tracklets')), (3, 1))
        missed.reset()
        self.assertEqual(missed.total('depth'), 0)

    def test_warnings_are_at_most_once_a_second(self) -> None:
        missed = MissedParts('test')
        missed.started()
        with self.assertLogs('modules.camera.capture_match', 'WARNING') as logs:
            for _ in range(5):
                missed.count(0, 'depth')
        self.assertEqual(len(logs.records), 1)


if __name__ == '__main__':
    unittest.main()
