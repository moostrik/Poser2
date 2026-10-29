"""PosePipeline — an app's pose pipeline, built once from its ``PipelineSpec``.

The builder resolves the spec at startup into direct callback chains, the same as hand wiring; nothing
walks the graph per frame. A stage runs its steps in order: the per-pose steps that follow each other in
one ``FilterTracker`` sized from the ids that reach the stage, a group step on the whole frame between
them (stamping the poses, writing the group record), and an async step splits the stage: the steps before it
run on the caller's thread, the steps after it on the thread the node emits on. The stage's synchronous steps
are timed as ``Pose <stage>``; the crossing's ``InterpolatorTracker`` as ``Pose interpolate``.

The group record travels with the poses: a source's stage starts a fresh one on every push, a stage that reads
another gets that stage's record, and the latest record is held across an async step and across the crossing,
where the tick re-stamps it. Frames leave a stage in one order: the app's callbacks (board, recorders, OSC),
the record's, then the stage's window tracker, then the stages that read it. ``update(domain, now)`` drives the
stages a tick enters: the interpolators of the crossing, then the tick sources polled at ``now``, their poses
merged, and the stage's steps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

from modules.utils import Broadcast, PerformanceTimer

from ..frame import FrameDict, FrameDictCallback, GroupRecord, GroupCallback, replace_group
from ..nodes import GroupNode
from ..trackers import FilterPipeline, FilterTracker, InterpolatorPipeline, InterpolatorTracker
from ..window import WindowTracker
from .spec import AsyncStep, Domain, FrameSource, GroupStep, Merge, PipelineSpec, PoseStep, StageSpec, TickSource

logger = logging.getLogger(__name__)

StageFunction = Callable[[FrameDict, GroupRecord], tuple[FrameDict, GroupRecord]]
StageCallback = Callable[[FrameDict, GroupRecord], None]


def _stage_name(stage: int) -> str:
    return getattr(stage, 'name', str(stage)).lower()


def _stamp(frames: FrameDict, fallback: float) -> float:
    """The newest stamp of the frames, or ``fallback`` when there are none."""
    return max((frame.timestamp for frame in frames.values()), default=fallback)


@dataclass
class _Stage:
    """One built stage: what reaches it, what it emits to, and what the tick drives."""
    spec:          StageSpec
    track_ids:     list[int]
    outputs:       Broadcast = field(default_factory=Broadcast)       # the app's frames callbacks
    group_outputs: Broadcast = field(default_factory=Broadcast)       # the app's group callbacks
    readers:       list[StageCallback] = field(default_factory=list)  # the stages that read this one
    window:        WindowTracker | None = None
    entry:         StageCallback | None = None                        # the first segment of the chain
    interpolators: InterpolatorTracker | None = None                  # the crossing, when the tick enters here
    sources:       list[TickSource] = field(default_factory=list)     # polled on the tick
    held_group:    GroupRecord = field(default_factory=GroupRecord)   # the latest record from across the crossing

    @property
    def name(self) -> str:
        return _stage_name(self.spec.stage)


class _AsyncSplit:
    """The hand-over at an async step: the frames go through the node, the group record waits for them."""

    def __init__(self, step: AsyncStep, then: StageCallback) -> None:
        self._node = step.node
        self._then = then
        self._group = GroupRecord()
        step.node.add_frames_callback(self._emitted)

    def submit(self, frames: FrameDict, group: GroupRecord) -> None:
        self._group = group
        self._node.submit(frames)

    def _emitted(self, frames: FrameDict) -> None:
        self._then(frames, self._group)


class PosePipeline:
    """The pipeline built from a ``PipelineSpec``; see the module docstring."""

    def __init__(self, spec: PipelineSpec) -> None:
        self._stages: dict[int, _Stage] = {}
        self._ticked: dict[Domain, list[_Stage]] = {}
        self._trackers: dict[int, list[FilterTracker]] = {}      # the per-pose steps of each stage, for reset()
        for stage_spec in spec.stages:
            self._build(stage_spec)

    # -- The app's side -------------------------------------------------------------

    @property
    def stages(self) -> list[int]:
        return list(self._stages)

    def track_ids(self, stage: int) -> list[int]:
        """The ids that reach ``stage``."""
        return list(self._stage(stage).track_ids)

    def add_frames_callback(self, stage: int, callback: FrameDictCallback) -> None:
        """Subscribe to a stage's poses; the callbacks run before the stage's window tracker and readers."""
        self._stage(stage).outputs.add_callback(callback)

    def add_group_callback(self, stage: int, callback: GroupCallback) -> None:
        """Subscribe to a stage's group record; the callbacks run right after the frames callbacks."""
        self._stage(stage).group_outputs.add_callback(callback)

    def window(self, stage: int) -> WindowTracker:
        """The stage's window tracker (a stage with window settings)."""
        window = self._stage(stage).window
        if window is None:
            raise ValueError(f"pose stage {_stage_name(stage)} has no window settings")
        return window

    def update(self, domain: Domain, now: float) -> None:
        """One tick of an output domain at ``now`` (perf_counter): the stages the tick enters, in order."""
        for stage in self._ticked.get(domain, ()):
            frames: FrameDict = stage.interpolators.update(now) if stage.interpolators is not None else {}
            if stage.sources:
                frames = self._merge(frames, stage.sources, now)
            assert stage.entry is not None
            stage.entry(frames, replace_group(stage.held_group, {}, timestamp=now))

    def reset(self) -> None:
        """Reset every stage's steps, crossing, held record and window."""
        for stage in self._stages.values():
            if stage.interpolators is not None:
                stage.interpolators.reset()
            if stage.window is not None:
                stage.window.reset()
            stage.held_group = GroupRecord()
            for step in stage.spec.steps:
                if isinstance(step, GroupStep):
                    step.node.reset()
            for tracker in self._trackers.get(stage.spec.stage, ()):
                tracker.reset()

    # -- The build --------------------------------------------------------------------

    def _stage(self, stage: int) -> _Stage:
        try:
            return self._stages[stage]
        except KeyError:
            raise ValueError(f"pose stage {_stage_name(stage)} is not in the pipeline") from None

    @staticmethod
    def _merge(frames: FrameDict, sources: list[TickSource], now: float) -> FrameDict:
        polled = [(source, source.update(now)) for source in sources]
        solo = [polled_frames for source, polled_frames in polled if source.solo]
        if solo:
            merged: FrameDict = {}
            for polled_frames in solo:
                merged.update(polled_frames)
            return merged
        merged = dict(frames)
        for _, polled_frames in polled:
            merged.update(polled_frames)
        return merged

    def _build(self, spec: StageSpec) -> None:
        name = _stage_name(spec.stage)
        if spec.stage in self._stages:
            raise ValueError(f"pose stage {name} is declared twice")
        input_stage, frame_source, sources = self._inputs(spec)

        # The ids that reach the stage: the union of its inputs', which must not overlap.
        track_ids: list[int] = []
        if input_stage is not None:
            track_ids.extend(input_stage.track_ids)
        if frame_source is not None:
            track_ids.extend(frame_source.track_ids)
        for source in sources:
            overlap = set(track_ids).intersection(source.track_ids)
            if overlap:
                raise ValueError(f"pose stage {name}: the ids {sorted(overlap)} reach it from two inputs")
            track_ids.extend(source.track_ids)
        if not track_ids:
            raise ValueError(f"pose stage {name} has no ids reaching it")

        stage = _Stage(spec, track_ids, sources=sources)
        if spec.window is not None:
            stage.window = WindowTracker(track_ids, spec.window, features=spec.window_features)

        crossing = input_stage is not None and input_stage.spec.domain != spec.domain
        if crossing:
            assert input_stage is not None
            stage.interpolators = InterpolatorTracker({
                tid: InterpolatorPipeline([step.build() for step in spec.interpolators]) for tid in input_stage.track_ids
            }, name="Pose interpolate")

            def cross(frames: FrameDict, group: GroupRecord, stage: _Stage = stage) -> None:
                assert stage.interpolators is not None
                stage.interpolators.set(frames)
                stage.held_group = group
            input_stage.readers.append(cross)

        stage.entry = self._chain(stage, track_ids)

        if frame_source is not None:
            def push(frames: FrameDict, stage: _Stage = stage) -> None:
                assert stage.entry is not None
                stage.entry(frames, GroupRecord(timestamp=_stamp(frames, 0.0)))
            frame_source.add_frames_callback(push)
        elif input_stage is not None and not crossing:
            input_stage.readers.append(stage.entry)
        if crossing or sources:
            self._ticked.setdefault(spec.domain, []).append(stage)
        self._stages[spec.stage] = stage

    def _inputs(self, spec: StageSpec) -> tuple[_Stage | None, FrameSource | None, list[TickSource]]:
        """The stage's declared inputs, checked against its domain."""
        name = _stage_name(spec.stage)
        inputs = list(spec.input.inputs) if isinstance(spec.input, Merge) else [spec.input]
        if isinstance(spec.input, Merge) and spec.domain == Domain.FRAME:
            raise ValueError(f"pose stage {name}: the frame domain takes one push source, not a merge")
        input_stage: _Stage | None = None
        frame_source: FrameSource | None = None
        sources: list[TickSource] = []
        for item in inputs:
            if isinstance(item, int):
                if item not in self._stages:
                    raise ValueError(f"pose stage {name} reads stage {_stage_name(item)}, which is not declared before it")
                if input_stage is not None:
                    raise ValueError(f"pose stage {name} reads two stages; a stage reads one")
                input_stage = self._stages[item]
            elif isinstance(item, TickSource):
                if spec.domain == Domain.FRAME:
                    raise ValueError(f"pose stage {name}: a tick source joins in an output domain, not the frame domain")
                sources.append(item)
            elif isinstance(item, FrameSource):
                if spec.domain != Domain.FRAME or len(inputs) > 1:
                    raise ValueError(f"pose stage {name}: a frame source is the one input of a frame-domain stage")
                frame_source = item
            else:
                raise ValueError(f"pose stage {name}: {item!r} is not a stage, a source or a merge")

        crossing = input_stage is not None and input_stage.spec.domain != spec.domain
        if crossing:
            assert input_stage is not None
            if input_stage.spec.domain != Domain.FRAME or spec.domain == Domain.FRAME:
                raise ValueError(f"pose stage {name} crosses from {input_stage.spec.domain.name} to {spec.domain.name}; "
                                 "the interpolators cross from the frame domain to an output domain only")
            if not spec.interpolators:
                raise ValueError(f"pose stage {name} crosses into {spec.domain.name} without interpolators")
        elif spec.interpolators:
            raise ValueError(f"pose stage {name} declares interpolators but does not cross a domain")
        if sources and input_stage is not None and not crossing:
            raise ValueError(f"pose stage {name}: a tick source joins at the stage the tick drives, "
                             "the crossing or a stage of sources only")
        if spec.domain != Domain.FRAME and any(isinstance(step, AsyncStep) for step in spec.steps):
            raise ValueError(f"pose stage {name}: an async step runs in the frame domain; an output domain runs on its tick")
        return input_stage, frame_source, sources

    def _chain(self, stage: _Stage, track_ids: list[int]) -> StageCallback:
        """The stage's steps as one callable, split at every async step; the tail emits the stage's frames and record."""
        timer = PerformanceTimer(name=f"Pose {stage.name}", sample_count=200)

        # The segments between the async steps, last first, so each one knows what comes after it.
        segments: list[list[StageFunction]] = [[]]
        asyncs: list[AsyncStep] = []
        run: list[PoseStep] = []

        def close_run() -> None:
            if run:
                tracker = FilterTracker({tid: FilterPipeline([step.build() for step in run]) for tid in track_ids})
                self._trackers.setdefault(stage.spec.stage, []).append(tracker)
                segments[-1].append(_per_pose(tracker))
                run.clear()

        for step in stage.spec.steps:
            if isinstance(step, PoseStep):
                run.append(step)
            elif isinstance(step, GroupStep):
                close_run()
                segments[-1].append(_group(step.node))
            elif isinstance(step, AsyncStep):
                close_run()
                asyncs.append(step)
                segments.append([])
            else:
                raise ValueError(f"pose stage {stage.name}: {step!r} is not a step")
        close_run()

        def emit(frames: FrameDict, group: GroupRecord) -> None:
            stage.outputs(frames)
            stage.group_outputs(group)
            if stage.window is not None:
                stage.window.process(frames)
            for reader in stage.readers:
                try:
                    reader(frames, group)
                except Exception:
                    logger.exception("Error in a stage reading pose stage %s", stage.name)

        def segment(functions: list[StageFunction], then: StageCallback) -> StageCallback:
            if not functions:
                return then

            def run_segment(frames: FrameDict, group: GroupRecord) -> None:
                start = perf_counter()
                for function in functions:
                    frames, group = function(frames, group)
                timer.add_time((perf_counter() - start) * 1000.0, report=False)
                then(frames, group)
            return run_segment

        then: StageCallback = segment(segments[-1], emit)
        for functions, step in zip(reversed(segments[:-1]), reversed(asyncs)):
            then = segment(functions, _AsyncSplit(step, then).submit)
        return then


def _per_pose(tracker: FilterTracker) -> StageFunction:
    def run(frames: FrameDict, group: GroupRecord) -> tuple[FrameDict, GroupRecord]:
        return tracker.process(frames), group
    return run


def _group(node: GroupNode) -> StageFunction:
    """The poses stamped first, so the record is written from the poses as the step left them."""
    def run(frames: FrameDict, group: GroupRecord) -> tuple[FrameDict, GroupRecord]:
        frames = node.process(frames)
        return frames, node.record(frames, group)
    return run
