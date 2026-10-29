"""The group similarity: the matrix from the rows, and the level by mode."""

import math
import unittest

import numpy as np

from modules.pose.analytics import GroupSimilarityMode, group_similarity_level, similarity_matrix
from modules.pose.features import Similarity

from ._builders import frame, scalar

ALL, ALL_BUT_ONE = GroupSimilarityMode.ALL, GroupSimilarityMode.ALL_BUT_ONE


def _matrix(*rows: list[float]) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


class SimilarityMatrixTest(unittest.TestCase):

    def test_a_pair_is_the_mean_of_both_directions(self) -> None:
        frames = {0: frame(0, features={Similarity: scalar(Similarity, {1: 0.8})}),
                  1: frame(1, features={Similarity: scalar(Similarity, {0: 0.6})})}
        ids, matrix = similarity_matrix(frames)
        self.assertEqual(ids, [0, 1])
        self.assertAlmostEqual(matrix[0, 1], 0.7)
        self.assertAlmostEqual(matrix[1, 0], 0.7)
        self.assertTrue(math.isnan(matrix[0, 0]))

    def test_one_direction_reads_alone_and_none_is_nan(self) -> None:
        frames = {0: frame(0, features={Similarity: scalar(Similarity, {1: 0.8})}),
                  1: frame(1),
                  2: frame(2)}
        _, matrix = similarity_matrix(frames)
        self.assertAlmostEqual(matrix[0, 1], 0.8)
        self.assertAlmostEqual(matrix[1, 0], 0.8)
        self.assertTrue(math.isnan(matrix[1, 2]))
        self.assertTrue(math.isnan(matrix[0, 2]))

    def test_an_id_past_the_row_reads_nan(self) -> None:
        frames = {0: frame(0, features={Similarity: scalar(Similarity, {1: 0.8})}), 9: frame(9)}
        ids, matrix = similarity_matrix(frames)
        self.assertEqual(ids, [0, 9])
        self.assertTrue(math.isnan(matrix[0, 1]))


class GroupSimilarityLevelTest(unittest.TestCase):

    def test_all_is_the_least_alike_pair(self) -> None:
        n = math.nan
        matrix = _matrix([n, 0.9, 0.5], [0.9, n, 0.7], [0.5, 0.7, n])
        self.assertAlmostEqual(group_similarity_level(matrix, ALL), 0.5)

    def test_a_pair_that_cannot_be_read_holds_all_at_nan(self) -> None:
        n = math.nan
        matrix = _matrix([n, 0.9, n], [0.9, n, 0.7], [n, 0.7, n])
        self.assertTrue(math.isnan(group_similarity_level(matrix, ALL)))

    def test_fewer_than_two_is_nan(self) -> None:
        self.assertTrue(math.isnan(group_similarity_level(np.full((1, 1), math.nan), ALL)))
        self.assertTrue(math.isnan(group_similarity_level(np.zeros((0, 0)), ALL_BUT_ONE)))

    def test_all_but_one_leaves_out_the_one_who_drags_the_group_down(self) -> None:
        n = math.nan
        # 0 and 1 are alike; 2 is far from both: without 2 the group reads 0.9.
        matrix = _matrix([n, 0.9, 0.2], [0.9, n, 0.3], [0.2, 0.3, n])
        self.assertAlmostEqual(group_similarity_level(matrix, ALL_BUT_ONE), 0.9)
        self.assertAlmostEqual(group_similarity_level(matrix, ALL), 0.2)

    def test_all_but_one_with_two_reads_as_all(self) -> None:
        n = math.nan
        matrix = _matrix([n, 0.4], [0.4, n])
        self.assertAlmostEqual(group_similarity_level(matrix, ALL_BUT_ONE), 0.4)

    def test_all_but_one_leaves_out_the_one_whose_pair_cannot_be_read(self) -> None:
        n = math.nan
        matrix = _matrix([n, 0.9, n], [0.9, n, 0.7], [n, 0.7, n])
        self.assertAlmostEqual(group_similarity_level(matrix, ALL_BUT_ONE), 0.9)

    def test_all_but_one_with_four_still_leaves_out_one(self) -> None:
        n = math.nan
        matrix = _matrix([n, 0.9, 0.8, 0.1], [0.9, n, 0.85, 0.1], [0.8, 0.85, n, 0.1], [0.1, 0.1, 0.1, n])
        self.assertAlmostEqual(group_similarity_level(matrix, ALL_BUT_ONE), 0.8)
        matrix = _matrix([n, 0.9, 0.1, 0.1], [0.9, n, 0.85, 0.1], [0.1, 0.85, n, 0.1], [0.1, 0.1, 0.1, n])
        self.assertAlmostEqual(group_similarity_level(matrix, ALL_BUT_ONE), 0.1)   # two out: one leave-out is not enough


if __name__ == "__main__":
    unittest.main()
