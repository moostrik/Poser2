"""Tests for the multi-track trackers: per-track isolation, reset on loss, failure isolation, and windowing."""

import math
import unittest

import numpy as np

from modules.pose.features import Age, Angles, BaseFeature
from modules.pose.frame import Frame, FrameDict
from modules.pose.nodes import AgeExtractor, AngleLerpInterpolator, FilterNode, InterpolatorNode, LerpInterpolatorSettings
from modules.pose.trackers import FilterPipeline, FilterTracker, InterpolatorPipeline, InterpolatorTracker
from modules.pose.window import WindowNode, WindowNodeSettings, WindowTracker

from ._builders import frame, scalar


class _RaiseFor(FilterNode):
    """Raises for one track id, passes every other frame through."""

    def __init__(self, track_id: int) -> None:
        self._track_id = track_id

    def process(self, pose: Frame) -> Frame:
        if pose.track_id == self._track_id:
            raise RuntimeError("boom")
        return pose


class _RaisingInterpolator(InterpolatorNode):
    """An interpolator that raises in set() or update() for one track id."""

    def __init__(self, track_id: int, raise_in: str) -> None:
        self._track_id = track_id
        self._raise_in = raise_in
        self._pose: Frame | None = None

    @property
    def feature_type(self) -> type[BaseFeature]:
        return Angles

    def set(self, pose: Frame | None) -> None:
        if self._raise_in == 'set' and pose is not None and pose.track_id == self._track_id:
            raise RuntimeError("boom")
        self._pose = pose

    def update(self) -> Frame | None:
        if self._raise_in == 'update' and self._pose is not None and self._pose.track_id == self._track_id:
            raise RuntimeError("boom")
        return self._pose

    def reset(self) -> None:
        self._pose = None

    def is_ready(self) -> bool:
        return self._pose is not None


def _age_tracker(n: int = 2) -> FilterTracker:
    return FilterTracker({i: FilterPipeline([AgeExtractor()]) for i in range(n)})


def _ages(out: FrameDict) -> dict[int, float]:
    return {tid: f[Age].value for tid, f in out.items()}


class FilterTrackerTest(unittest.TestCase):
    def test_tracks_keep_independent_state(self) -> None:
        tracker = _age_tracker()
        tracker.process({0: frame(0, 0.0), 1: frame(1, 5.0)})
        out = tracker.process({0: frame(0, 1.0), 1: frame(1, 7.0)})
        self.assertAlmostEqual(out[0][Age].value, 1.0)
        self.assertAlmostEqual(out[1][Age].value, 2.0)

    def test_missing_track_is_reset(self) -> None:
        tracker = _age_tracker()
        tracker.process({0: frame(0, 0.0), 1: frame(1, 0.0)})
        tracker.process({0: frame(0, 1.0), 1: frame(1, 1.0)})
        tracker.process({1: frame(1, 2.0)})                       # track 0 lost
        tracker.process({0: frame(0, 10.0), 1: frame(1, 10.0)})
        out = tracker.process({0: frame(0, 11.0), 1: frame(1, 11.0)})
        self.assertAlmostEqual(out[0][Age].value, 1.0)
        self.assertAlmostEqual(out[1][Age].value, 11.0)

    def test_failing_track_passes_through_and_others_still_process(self) -> None:
        tracker = FilterTracker({i: FilterPipeline([_RaiseFor(0), AgeExtractor()]) for i in range(3)})
        raw = frame(0, 1.0)
        with self.assertLogs('modules.pose.trackers', level='ERROR'):
            tracker.process({0: frame(0, 0.0), 1: frame(1, 0.0), 2: frame(2, 0.0)})
            out = tracker.process({0: raw, 1: frame(1, 1.0), 2: frame(2, 1.0)})
        self.assertIs(out[0], raw)
        self.assertAlmostEqual(out[1][Age].value, 1.0)
        self.assertAlmostEqual(out[2][Age].value, 1.0)

    def test_callbacks_receive_the_returned_dict(self) -> None:
        tracker = _age_tracker()
        received: list[FrameDict] = []
        tracker.add_frames_callback(lambda frames: received.append(frames))
        out = tracker.process({0: frame(0, 0.0)})
        self.assertEqual(len(received), 1)
        self.assertIs(received[0], out)

    def test_failing_callback_does_not_break_others(self) -> None:
        tracker = _age_tracker()
        received: list[FrameDict] = []

        def bad(_frames: FrameDict) -> None:
            raise RuntimeError("boom")

        tracker.add_frames_callback(bad)
        tracker.add_frames_callback(lambda frames: received.append(frames))
        with self.assertLogs('modules.pose.frame', level='ERROR'):
            tracker.process({0: frame(0, 0.0)})
        self.assertEqual(len(received), 1)

    def test_empty_pipelines_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FilterTracker({})
        with self.assertRaises(ValueError):
            FilterPipeline([])


def _angle(track_id: int, head: float) -> Frame:
    return frame(track_id, 0.0, features={Angles: scalar(Angles, {0: head})})


class InterpolatorTrackerTest(unittest.TestCase):
    def _tracker(self) -> InterpolatorTracker:
        return InterpolatorTracker({i: InterpolatorPipeline([AngleLerpInterpolator(LerpInterpolatorSettings())]) for i in range(2)})

    def test_interpolates_each_present_track(self) -> None:
        tracker = self._tracker()
        tracker.set({0: _angle(0, 0.1), 1: _angle(1, 0.2)})
        out = tracker.update()
        self.assertAlmostEqual(out[0][Angles][0], 0.1, places=5)
        self.assertAlmostEqual(out[1][Angles][0], 0.2, places=5)

    def test_missing_track_is_reset_and_dropped(self) -> None:
        tracker = self._tracker()
        tracker.set({0: _angle(0, 0.1), 1: _angle(1, 0.2)})
        tracker.update()
        tracker.set({0: _angle(0, 0.1)})
        self.assertEqual(set(tracker.update()), {0})

    def test_callbacks_receive_the_returned_dict(self) -> None:
        tracker = self._tracker()
        received: list[FrameDict] = []
        tracker.add_frames_callback(lambda frames: received.append(frames))
        tracker.set({0: _angle(0, 0.1)})
        out = tracker.update()
        self.assertIs(received[0], out)

    @unittest.expectedFailure
    def test_failing_set_does_not_skip_later_tracks(self) -> None:
        # One try wraps the whole per-track loop in set(), so a raise for track 0 skips track 1.
        tracker = InterpolatorTracker({i: InterpolatorPipeline([_RaisingInterpolator(0, 'set')]) for i in range(2)})
        with self.assertLogs('modules.pose.trackers', level='ERROR'):
            tracker.set({0: _angle(0, 0.1), 1: _angle(1, 0.2)})
        self.assertIn(1, tracker.update())

    @unittest.expectedFailure
    def test_failing_update_does_not_drop_later_tracks(self) -> None:
        # One try wraps the whole per-track loop in update(), so a raise for track 0 drops track 1.
        tracker = InterpolatorTracker({i: InterpolatorPipeline([_RaisingInterpolator(0, 'update')]) for i in range(2)})
        tracker.set({0: _angle(0, 0.1), 1: _angle(1, 0.2)})
        with self.assertLogs('modules.pose.trackers', level='ERROR'):
            out = tracker.update()
        self.assertIn(1, out)


def _age_frame(value: float, track_id: int = 0) -> Frame:
    return frame(track_id, 0.0, features={Age: Age.from_value(value)})


def _window_settings(size: int, emit_partial: bool = True) -> WindowNodeSettings:
    settings = WindowNodeSettings()
    settings.window_size = size
    settings.emit_partial = emit_partial
    return settings


class WindowNodeTest(unittest.TestCase):
    def test_partial_window_is_full_size_with_unfilled_slots_masked_oldest_first(self) -> None:
        node = WindowNode(Age, _window_settings(3))
        node.process(_age_frame(1.0))
        window = node.process(_age_frame(2.0))
        self.assertEqual(window.shape, (3, 1))
        np.testing.assert_array_equal(window.mask[:, 0], [False, True, True])
        np.testing.assert_allclose(window.values[1:, 0], [1.0, 2.0])

    def test_wrapped_ring_buffer_is_oldest_first(self) -> None:
        node = WindowNode(Age, _window_settings(3))
        for v in (1.0, 2.0, 3.0, 4.0):
            window = node.process(_age_frame(v))
        np.testing.assert_allclose(window.values[:, 0], [2.0, 3.0, 4.0])
        self.assertTrue(np.all(window.mask))

    def test_missing_value_is_masked(self) -> None:
        node = WindowNode(Age, _window_settings(2))
        node.process(_age_frame(1.0))
        window = node.process(frame())
        np.testing.assert_array_equal(window.mask[:, 0], [True, False])
        self.assertEqual(window.values[1, 0], 0.0)

    def test_no_partial_emission_until_full(self) -> None:
        node = WindowNode(Age, _window_settings(3, emit_partial=False))
        self.assertIsNone(node.process(_age_frame(1.0)))
        self.assertIsNone(node.process(_age_frame(2.0)))
        self.assertIsNotNone(node.process(_age_frame(3.0)))

    def test_emitted_window_is_a_copy(self) -> None:
        node = WindowNode(Age, _window_settings(2))
        node.process(_age_frame(1.0))
        window = node.process(_age_frame(2.0))
        node.process(_age_frame(3.0))
        np.testing.assert_allclose(window.values[:, 0], [1.0, 2.0])

    def test_window_carries_feature_metadata(self) -> None:
        window = WindowNode(Angles, _window_settings(2)).process(frame())
        self.assertEqual(window.feature_len, len(Angles.enum()))
        self.assertEqual(window.range, Angles.range())

    def test_resize_reallocates_and_clears(self) -> None:
        settings = _window_settings(3)
        node = WindowNode(Age, settings)
        node.process(_age_frame(1.0))
        settings.window_size = 5
        window = node.process(_age_frame(2.0))
        self.assertEqual(window.shape, (5, 1))
        self.assertEqual(int(window.mask.sum()), 1)

    def test_reset_clears(self) -> None:
        node = WindowNode(Age, _window_settings(3))
        node.process(_age_frame(1.0))
        node.reset()
        self.assertEqual(int(node.process(_age_frame(2.0)).mask.sum()), 1)


class WindowTrackerTest(unittest.TestCase):
    def test_emits_per_feature_per_track_windows(self) -> None:
        tracker = WindowTracker(2, _window_settings(3), features=[Age])
        received: list = []
        tracker.add_windows_callback(lambda windows: received.append(windows))
        tracker.process({0: _age_frame(1.0, 0), 1: _age_frame(2.0, 1)})
        self.assertEqual(set(received[0]), {Age})
        self.assertEqual(set(received[0][Age]), {0, 1})

    def test_missing_track_window_is_reset(self) -> None:
        tracker = WindowTracker(2, _window_settings(3), features=[Age])
        received: list = []
        tracker.add_windows_callback(lambda windows: received.append(windows))
        tracker.process({0: _age_frame(1.0, 0), 1: _age_frame(1.0, 1)})
        tracker.process({0: _age_frame(2.0, 0)})
        tracker.process({0: _age_frame(3.0, 0), 1: _age_frame(3.0, 1)})
        last = received[-1][Age]
        self.assertEqual(int(last[0].mask.sum()), 3)
        self.assertEqual(int(last[1].mask.sum()), 1)
        self.assertNotIn(1, received[1][Age])

    def test_defaults_to_all_scalar_features(self) -> None:
        tracker = WindowTracker(1, _window_settings(2))
        received: list = []
        tracker.add_windows_callback(lambda windows: received.append(windows))
        tracker.process({0: frame()})
        self.assertIn(Angles, received[0])
        self.assertIn(Age, received[0])
        self.assertTrue(math.isfinite(received[0][Angles][0].values.sum()))


if __name__ == "__main__":
    unittest.main()
