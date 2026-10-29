"""The pipeline builder: what a spec becomes, in what order frames move, and what it refuses to build."""

import unittest
from enum import IntEnum
from functools import partial

from modules.pose.features import Age, BaseFeature
from modules.pose.frame import Frame, FrameDict, replace
from modules.pose.nodes import FilterNode, GroupNode, InterpolatorNode
from modules.pose.pipeline import AsyncStep, Domain, GroupStep, Merge, PipelineSpec, PosePipeline, PoseStep, StageSpec
from modules.pose.window import WindowNodeSettings

from ._builders import frame


class S(IntEnum):
    RAW = 0
    CLEAN = 1
    LERP = 2
    AFTER = 3


class _Source:
    """A frame source: pushes what it is given to the stage that reads it."""

    def __init__(self, track_ids: list[int]) -> None:
        self.track_ids = track_ids
        self._callbacks: list = []

    def add_frames_callback(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, frames: FrameDict) -> None:
        for callback in self._callbacks:
            callback(frames)


class _TickSource:
    """A tick source: one frame per id, stamped with the tick it was polled at."""

    def __init__(self, track_ids: list[int], solo: bool = False) -> None:
        self.track_ids = track_ids
        self.solo = solo
        self.polled: list[float] = []

    def update(self, now: float) -> FrameDict:
        self.polled.append(now)
        return {tid: frame(track_id=tid, t=now) for tid in self.track_ids}


class _Async:
    """An async node that holds what it gets until ``release``."""

    def __init__(self) -> None:
        self.submitted: list[FrameDict] = []
        self._callbacks: list = []

    def submit(self, frames: FrameDict) -> None:
        self.submitted.append(frames)

    def add_frames_callback(self, callback) -> None:
        self._callbacks.append(callback)

    def release(self) -> None:
        frames = self.submitted[-1]
        for callback in self._callbacks:
            callback(frames)


class _Tag(FilterNode):
    """A per-pose step that logs its tag with the id it ran on, and counts its resets."""

    def __init__(self, tag: str, log: list) -> None:
        self.tag = tag
        self.log = log
        self.resets = 0

    def process(self, pose: Frame) -> Frame:
        self.log.append((self.tag, pose.track_id, id(self)))
        return replace(pose, {Age: Age.from_value(1.0)})

    def reset(self) -> None:
        self.resets += 1


class _Group(GroupNode):
    def __init__(self, log: list) -> None:
        self.log = log
        self.resets = 0

    def process(self, frames: FrameDict) -> FrameDict:
        self.log.append(('group', tuple(sorted(frames)), id(self)))
        return frames

    def reset(self) -> None:
        self.resets += 1


class _Hold(InterpolatorNode):
    """An interpolator that hands back the last target it was set."""

    def __init__(self) -> None:
        self._pose: Frame | None = None

    def set(self, pose: Frame | None) -> None:
        self._pose = pose

    def update(self) -> Frame | None:
        return self._pose

    def reset(self) -> None:
        self._pose = None

    def is_ready(self) -> bool:
        return True

    @property
    def feature_type(self) -> type[BaseFeature]:
        return Age


def _frames(*ids: int, t: float = 0.0) -> FrameDict:
    return {tid: frame(track_id=tid, t=t) for tid in ids}


class _Collect:
    """Records every stage's output in order of arrival."""

    def __init__(self) -> None:
        self.events: list[tuple[str, set[int]]] = []

    def __call__(self, name: str, frames: FrameDict) -> None:
        self.events.append((name, set(frames)))


class BuildTest(unittest.TestCase):

    def test_steps_run_in_order_per_pose_then_group_then_per_pose(self) -> None:
        log: list = []
        source = _Source([0, 1])
        pipeline = PosePipeline(PipelineSpec([
            StageSpec(S.RAW, source, [PoseStep(_Tag, 'a', log), GroupStep(_Group(log)), PoseStep(_Tag, 'b', log)]),
        ]))
        out = _Collect()
        pipeline.add_frames_callback(S.RAW, partial(out, 'raw'))
        source.emit(_frames(0, 1))
        self.assertEqual([(tag, tid) for tag, tid, _ in log], [('a', 0), ('a', 1), ('group', (0, 1)), ('b', 0), ('b', 1)])
        self.assertEqual(out.events, [('raw', {0, 1})])

    def test_a_pose_step_is_one_instance_per_id(self) -> None:
        log: list = []
        source = _Source([0, 1, 2])
        PosePipeline(PipelineSpec([StageSpec(S.RAW, source, [PoseStep(_Tag, 'a', log)])]))
        source.emit(_frames(0, 1, 2))
        source.emit(_frames(0, 1, 2))
        instances = {tid: {inst for _, t, inst in log if t == tid} for tid in (0, 1, 2)}
        self.assertTrue(all(len(v) == 1 for v in instances.values()))
        self.assertEqual(len(set.union(*instances.values())), 3)

    def test_a_stage_reads_the_one_before_it(self) -> None:
        log: list = []
        source = _Source([0])
        pipeline = PosePipeline(PipelineSpec([
            StageSpec(S.RAW, source, [PoseStep(_Tag, 'raw', log)]),
            StageSpec(S.CLEAN, S.RAW, [PoseStep(_Tag, 'clean', log)]),
        ]))
        out = _Collect()
        pipeline.add_frames_callback(S.RAW, partial(out, 'raw'))
        pipeline.add_frames_callback(S.CLEAN, partial(out, 'clean'))
        source.emit(_frames(0))
        self.assertEqual([tag for tag, _, _ in log], ['raw', 'clean'])
        self.assertEqual(out.events, [('raw', {0}), ('clean', {0})])
        self.assertEqual(pipeline.stages, [S.RAW, S.CLEAN])
        self.assertEqual(pipeline.track_ids(S.CLEAN), [0])

    def test_frames_leave_a_stage_to_the_app_then_the_window_then_the_readers(self) -> None:
        order: list[str] = []
        source = _Source([0])
        pipeline = PosePipeline(PipelineSpec([
            StageSpec(S.RAW, source, [], window=WindowNodeSettings()),
            StageSpec(S.CLEAN, S.RAW, []),
        ]))
        pipeline.add_frames_callback(S.CLEAN, lambda _f: order.append('reader'))
        pipeline.window(S.RAW).add_windows_callback(lambda _w: order.append('window'))
        pipeline.add_frames_callback(S.RAW, lambda _f: order.append('app'))
        source.emit({0: frame(track_id=0, features={Age: Age.from_value(1.0)})})
        self.assertEqual(order, ['app', 'window', 'reader'])

    def test_an_async_step_splits_the_stage_at_its_thread(self) -> None:
        log: list = []
        source = _Source([0])
        node = _Async()
        pipeline = PosePipeline(PipelineSpec([
            StageSpec(S.RAW, source, [PoseStep(_Tag, 'before', log), AsyncStep(node), PoseStep(_Tag, 'after', log)]),
        ]))
        out = _Collect()
        pipeline.add_frames_callback(S.RAW, partial(out, 'raw'))
        source.emit(_frames(0))
        self.assertEqual([tag for tag, _, _ in log], ['before'])
        self.assertEqual(len(node.submitted), 1)
        self.assertIn(Age, node.submitted[0][0])          # the steps before it ran on what it got
        self.assertEqual(out.events, [])
        node.release()
        self.assertEqual([tag for tag, _, _ in log], ['before', 'after'])
        self.assertEqual(out.events, [('raw', {0})])

    def test_the_tick_enters_at_the_crossing_and_polls_the_sources(self) -> None:
        source = _Source([0, 1])
        dummy = _TickSource([5])
        pipeline = PosePipeline(PipelineSpec([
            StageSpec(S.RAW, source, []),
            StageSpec(S.LERP, Merge(S.RAW, dummy), [], domain=Domain.OUTPUT, interpolators=[PoseStep(_Hold)]),
            StageSpec(S.AFTER, S.LERP, [], domain=Domain.OUTPUT),
        ]))
        out = _Collect()
        pipeline.add_frames_callback(S.LERP, partial(out, 'lerp'))
        pipeline.add_frames_callback(S.AFTER, partial(out, 'after'))
        self.assertEqual(pipeline.track_ids(S.LERP), [0, 1, 5])
        pipeline.update(Domain.OUTPUT, 7.0)                   # before any frame: the dummy alone
        self.assertEqual(out.events, [('lerp', {5}), ('after', {5})])
        source.emit(_frames(0, 1))
        self.assertEqual(len(out.events), 2)                  # the frame domain does not run the output stages
        pipeline.update(Domain.OUTPUT, 8.0)
        self.assertEqual(out.events[2:], [('lerp', {0, 1, 5}), ('after', {0, 1, 5})])
        self.assertEqual(dummy.polled, [7.0, 8.0])
        pipeline.update(Domain.FRAME, 9.0)                    # nothing enters the frame domain on a tick
        self.assertEqual(len(out.events), 4)

    def test_a_solo_source_keeps_the_others_out(self) -> None:
        source = _Source([0])
        dummy = _TickSource([5])
        pipeline = PosePipeline(PipelineSpec([
            StageSpec(S.RAW, source, []),
            StageSpec(S.LERP, Merge(S.RAW, dummy), [], domain=Domain.OUTPUT, interpolators=[PoseStep(_Hold)]),
        ]))
        out = _Collect()
        pipeline.add_frames_callback(S.LERP, partial(out, 'lerp'))
        source.emit(_frames(0))
        pipeline.update(Domain.OUTPUT, 1.0)
        dummy.solo = True
        pipeline.update(Domain.OUTPUT, 2.0)
        self.assertEqual(out.events, [('lerp', {0, 5}), ('lerp', {5})])

    def test_reset_reaches_every_step(self) -> None:
        log: list = []
        source = _Source([0, 1])
        group = _Group(log)
        pipeline = PosePipeline(PipelineSpec([
            StageSpec(S.RAW, source, [PoseStep(_Tag, 'a', log), GroupStep(group)]),
            StageSpec(S.LERP, S.RAW, [], domain=Domain.OUTPUT, interpolators=[PoseStep(_Hold)]),
        ]))
        out = _Collect()
        pipeline.add_frames_callback(S.LERP, partial(out, 'lerp'))
        source.emit(_frames(0, 1))
        pipeline.reset()
        pipeline.update(Domain.OUTPUT, 1.0)
        self.assertEqual(group.resets, 1)
        self.assertEqual(out.events, [('lerp', set())])        # the interpolators forgot their targets


class RefusalTest(unittest.TestCase):

    def _refuses(self, *stages: StageSpec) -> None:
        with self.assertRaises(ValueError):
            PosePipeline(PipelineSpec(list(stages)))

    def test_an_input_stage_must_be_declared_before(self) -> None:
        self._refuses(StageSpec(S.CLEAN, S.RAW, []))

    def test_a_stage_is_declared_once(self) -> None:
        source = _Source([0])
        self._refuses(StageSpec(S.RAW, source, []), StageSpec(S.RAW, source, []))

    def test_a_crossing_needs_interpolators(self) -> None:
        self._refuses(StageSpec(S.RAW, _Source([0]), []), StageSpec(S.LERP, S.RAW, [], domain=Domain.OUTPUT))

    def test_interpolators_cross_a_domain(self) -> None:
        self._refuses(StageSpec(S.RAW, _Source([0]), []), StageSpec(S.CLEAN, S.RAW, [], interpolators=[PoseStep(_Hold)]))

    def test_the_frame_domain_takes_one_push_source(self) -> None:
        self._refuses(StageSpec(S.RAW, Merge(_Source([0]), _TickSource([5])), []))
        self._refuses(StageSpec(S.RAW, _TickSource([5]), []))
        self._refuses(StageSpec(S.RAW, _Source([0]), [], domain=Domain.OUTPUT))

    def test_ids_reach_a_merge_from_one_input_each(self) -> None:
        self._refuses(StageSpec(S.RAW, _Source([0, 1]), []),
                      StageSpec(S.LERP, Merge(S.RAW, _TickSource([1])), [], domain=Domain.OUTPUT, interpolators=[PoseStep(_Hold)]))

    def test_a_tick_source_joins_where_the_tick_enters(self) -> None:
        self._refuses(StageSpec(S.RAW, _Source([0]), []),
                      StageSpec(S.LERP, S.RAW, [], domain=Domain.OUTPUT, interpolators=[PoseStep(_Hold)]),
                      StageSpec(S.AFTER, Merge(S.LERP, _TickSource([5])), [], domain=Domain.OUTPUT))

    def test_an_async_step_runs_in_the_frame_domain(self) -> None:
        self._refuses(StageSpec(S.RAW, _Source([0]), []),
                      StageSpec(S.LERP, S.RAW, [AsyncStep(_Async())], domain=Domain.OUTPUT, interpolators=[PoseStep(_Hold)]))

    def test_a_window_is_only_where_it_is_declared(self) -> None:
        pipeline = PosePipeline(PipelineSpec([StageSpec(S.RAW, _Source([0]), [])]))
        with self.assertRaises(ValueError):
            pipeline.window(S.RAW)
        with self.assertRaises(ValueError):
            pipeline.window(S.CLEAN)


if __name__ == "__main__":
    unittest.main()
