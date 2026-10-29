"""The group record: the Frame contract without an identity, and its slot on the board."""

import math
import unittest

from modules.board import GroupStoreMixin
from modules.pose.features import Age, MotionTime
from modules.pose.frame import GroupRecord, replace_group


class GroupRecordTest(unittest.TestCase):

    def test_an_unwritten_feature_is_a_nan_dummy_and_never_raises(self) -> None:
        group = GroupRecord()
        self.assertTrue(math.isnan(group[Age].value))
        self.assertEqual(group[Age].score, 0.0)
        self.assertNotIn(Age, group)
        self.assertEqual(len(group), 0)

    def test_replace_merges_and_keeps_the_stamp_unless_given_one(self) -> None:
        group = replace_group(GroupRecord(timestamp=3.0), {Age: Age.from_value(1.0)})
        self.assertEqual(group.timestamp, 3.0)
        self.assertIn(Age, group)
        later = replace_group(group, {MotionTime: MotionTime.from_value(2.0)}, timestamp=4.0)
        self.assertEqual(later.timestamp, 4.0)
        self.assertEqual(later[Age].value, 1.0)
        self.assertEqual(later[MotionTime].value, 2.0)
        self.assertNotIn(MotionTime, group)                   # the source record is untouched

    def test_a_record_is_immutable(self) -> None:
        group = GroupRecord()
        with self.assertRaises(AttributeError):
            group.timestamp = 1.0                              # type: ignore[misc]
        with self.assertRaises(AttributeError):
            del group._features


class GroupStoreTest(unittest.TestCase):

    def test_a_stage_never_written_reads_an_empty_record(self) -> None:
        board = GroupStoreMixin()
        self.assertEqual(len(board.get_group(2)), 0)
        self.assertTrue(math.isnan(board.get_group(2)[Age].value))

    def test_the_latest_record_per_stage(self) -> None:
        board = GroupStoreMixin()
        board.set_group(1, replace_group(GroupRecord(), {Age: Age.from_value(1.0)}))
        board.set_group(1, replace_group(GroupRecord(), {Age: Age.from_value(2.0)}))
        board.set_group(2, GroupRecord(timestamp=5.0))
        self.assertEqual(board.get_group(1)[Age].value, 2.0)
        self.assertEqual(board.get_group(2).timestamp, 5.0)


if __name__ == "__main__":
    unittest.main()
