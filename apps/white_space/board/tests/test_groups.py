"""The group records' slot on the board: the latest record per stage, an empty one for a stage never written."""

import math
import unittest

from modules.pose.features import Age
from modules.pose.frame import GroupRecord, replace_group

from .. import GroupStoreMixin


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
