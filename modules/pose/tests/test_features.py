"""Tests for the feature base-class contract, swept over every registered feature, plus feature-specific maths."""

import math
import unittest

import numpy as np

from modules.pose.features import (
    FEATURES, AggregationMethod, AngleLandmark, Angles, AngleSymmetry, BBox, BBoxElement, LeaderScore,
    MotionGate, PointLandmark, Points2D, BaseVectorFeature,
)
from modules.utils import Rect

from ._builders import points, scalar

# Track-indexed features whose dummy is zeros, not NaN — see test_track_feature_dummies_are_nan.
_ZERO_DUMMIES = {LeaderScore, MotionGate}


def _in_range_value(feature_type: type) -> float:
    lo, hi = feature_type.range()
    if math.isinf(lo) or math.isinf(hi):
        return 0.5
    return (lo + hi) / 2.0


def _filled(feature_type: type, value: float, score: float = 1.0):
    dummy = feature_type.create_dummy()
    values = np.full(dummy.values.shape, value, dtype=np.float32)
    scores = np.full(dummy.scores.shape, score, dtype=np.float32)
    return feature_type(values, scores), values, scores


class FeatureContractTest(unittest.TestCase):
    """Every feature in FEATURES — a new feature gets these checks by being registered."""

    def test_dummy_has_zero_scores_and_validates(self) -> None:
        for ft in FEATURES:
            with self.subTest(feature=ft.__name__):
                dummy = ft.create_dummy()
                self.assertIs(type(dummy), ft)
                self.assertEqual(len(dummy), ft.length())
                self.assertTrue(np.all(dummy.scores == 0.0))
                ok, err = dummy.validate()
                self.assertTrue(ok, err)

    def test_dummy_values_are_nan(self) -> None:
        for ft in FEATURES:
            if ft in _ZERO_DUMMIES:
                continue
            with self.subTest(feature=ft.__name__):
                self.assertTrue(np.all(np.isnan(ft.create_dummy().values)))

    @unittest.expectedFailure
    def test_track_feature_dummies_are_nan(self) -> None:
        # Pose Frame contract: missing data is NaN with score 0. LeaderScore and MotionGate dummies are
        # zeros, so a reader can't tell "not computed" from "computed as 0" without checking scores.
        for ft in _ZERO_DUMMIES:
            self.assertTrue(np.all(np.isnan(ft.create_dummy().values)), ft.__name__)

    def test_dummy_arrays_are_read_only(self) -> None:
        for ft in FEATURES:
            with self.subTest(feature=ft.__name__):
                dummy = ft.create_dummy()
                self.assertFalse(dummy.values.flags.writeable)
                self.assertFalse(dummy.scores.flags.writeable)

    def test_construction_takes_ownership_and_tracks_validity(self) -> None:
        for ft in FEATURES:
            with self.subTest(feature=ft.__name__):
                feature, values, scores = _filled(ft, _in_range_value(ft))
                self.assertFalse(values.flags.writeable)
                self.assertFalse(scores.flags.writeable)
                self.assertEqual(feature.valid_count, ft.length())
                self.assertTrue(np.all(feature.valid_mask))
                ok, err = feature.validate()
                self.assertTrue(ok, err)

    def test_validate_flags_nan_with_nonzero_score(self) -> None:
        for ft in FEATURES:
            with self.subTest(feature=ft.__name__):
                dummy = ft.create_dummy()
                values = np.full(dummy.values.shape, _in_range_value(ft), dtype=np.float32)
                values[0] = np.nan
                scores = np.ones(dummy.scores.shape, dtype=np.float32)
                feature = ft(values, scores)
                self.assertEqual(feature.valid_count, ft.length() - 1)
                self.assertFalse(bool(feature.valid_mask[0]))
                ok, _ = feature.validate(check_ranges=False)
                self.assertFalse(ok)

    def test_validate_flags_out_of_range_values(self) -> None:
        for ft in FEATURES:
            _, hi = ft.range()
            if math.isinf(hi):
                continue
            with self.subTest(feature=ft.__name__):
                feature, _, _ = _filled(ft, hi + 1.0)
                self.assertFalse(feature.validate(check_ranges=True)[0])
                self.assertTrue(feature.validate(check_ranges=False)[0])

    def test_vector_element_is_invalid_when_any_component_is_nan(self) -> None:
        for ft in FEATURES:
            if not issubclass(ft, BaseVectorFeature):
                continue
            with self.subTest(feature=ft.__name__):
                dummy = ft.create_dummy()
                values = np.full(dummy.values.shape, _in_range_value(ft), dtype=np.float32)
                values[0, 0] = np.nan
                self.assertFalse(bool(ft(values, np.ones(ft.length(), dtype=np.float32)).valid_mask[0]))


class AnglesTest(unittest.TestCase):
    def test_subtract_wraps_across_pi(self) -> None:
        a = scalar(Angles, {AngleLandmark.head: math.pi - 0.1})
        b = scalar(Angles, {AngleLandmark.head: -math.pi + 0.1})
        self.assertAlmostEqual(a.subtract(b)[AngleLandmark.head], -0.2, places=5)
        self.assertAlmostEqual(b.subtract(a)[AngleLandmark.head], 0.2, places=5)

    def test_subtract_takes_minimum_score(self) -> None:
        a = scalar(Angles, {AngleLandmark.left_knee: 0.5}, score=0.9)
        b = scalar(Angles, {AngleLandmark.left_knee: 0.2}, score=0.4)
        self.assertAlmostEqual(a.subtract(b).get_score(AngleLandmark.left_knee), 0.4, places=6)

    def test_subtract_with_missing_side_is_nan_with_zero_score(self) -> None:
        a = scalar(Angles, {AngleLandmark.left_knee: 0.5})
        diff = a.subtract(Angles.create_dummy())
        self.assertTrue(math.isnan(diff[AngleLandmark.left_knee]))
        self.assertEqual(diff.get_score(AngleLandmark.left_knee), 0.0)

    def test_get_fill_replaces_nan(self) -> None:
        a = scalar(Angles, {AngleLandmark.head: 0.3})
        self.assertEqual(a.get(AngleLandmark.left_knee, fill=-1.0), -1.0)
        self.assertAlmostEqual(a.get(AngleLandmark.head, fill=-1.0), 0.3, places=6)


class AggregateTest(unittest.TestCase):
    """NormalizedScalarFeature statistics, on AngleSymmetry (a fixed-length normalised feature)."""

    def _sym(self, values: list[float], scores: list[float] | None = None) -> AngleSymmetry:
        v = np.array(values, dtype=np.float32)
        s = np.array(scores if scores is not None else [0.0 if math.isnan(x) else 1.0 for x in values], dtype=np.float32)
        return AngleSymmetry(v, s)

    def test_mean_and_harmonic_mean(self) -> None:
        self.assertEqual(AngleSymmetry.length(), 4)
        sym = self._sym([0.9, 0.9, 0.9, 0.3])
        self.assertAlmostEqual(sym.aggregate(AggregationMethod.MEAN), 0.75, places=5)
        self.assertAlmostEqual(sym.aggregate(AggregationMethod.HARMONIC_MEAN), 4.0 / (3 / 0.9 + 1 / 0.3), places=5)

    def test_nan_elements_are_ignored(self) -> None:
        sym = self._sym([0.2, math.nan, 0.6, math.nan])
        self.assertAlmostEqual(sym.aggregate(AggregationMethod.MEAN), 0.4, places=5)

    def test_min_confidence_filters_elements(self) -> None:
        sym = self._sym([0.2, 0.4, 0.6, 0.8], scores=[0.1, 0.1, 0.9, 0.9])
        self.assertAlmostEqual(sym.aggregate(AggregationMethod.MEAN, min_confidence=0.5), 0.7, places=5)

    def test_zero_value_collapses_harmonic_mean(self) -> None:
        sym = self._sym([1.0, 1.0, 1.0, 0.0])
        self.assertLess(sym.aggregate(AggregationMethod.HARMONIC_MEAN), 1e-3)

    def test_nothing_qualifying_is_nan(self) -> None:
        sym = self._sym([math.nan] * 4)
        for method in AggregationMethod:
            with self.subTest(method=method.name):
                self.assertTrue(math.isnan(sym.aggregate(method)))


class VectorAndBoxTest(unittest.TestCase):
    def test_points_accessors(self) -> None:
        pts = points({PointLandmark.nose: (0.25, 0.75)})
        self.assertEqual(pts.get(PointLandmark.nose), (0.25, 0.75))
        self.assertTrue(math.isnan(pts.get_x(PointLandmark.left_eye)))
        self.assertEqual(pts.get_y(PointLandmark.left_eye, fill=0.0), 0.0)
        self.assertEqual(pts.valid_count, 1)

    def test_bbox_rect_round_trip(self) -> None:
        box = BBox.from_rect(Rect(0.2, 0.1, 0.4, 0.8))
        self.assertAlmostEqual(box[BBoxElement.centre_x], 0.4, places=6)
        self.assertAlmostEqual(box[BBoxElement.centre_y], 0.5, places=6)
        rect = box.to_rect()
        self.assertAlmostEqual(rect.x, 0.2, places=6)
        self.assertAlmostEqual(rect.height, 0.8, places=6)

    def test_bbox_zero_size_becomes_nan(self) -> None:
        box = BBox.from_rect(Rect(0.2, 0.1, 0.0, 0.8))
        self.assertTrue(math.isnan(box[BBoxElement.width]))
        self.assertEqual(box.get_score(BBoxElement.width), 0.0)


if __name__ == "__main__":
    unittest.main()
