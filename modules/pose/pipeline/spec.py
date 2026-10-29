"""The declarations an app's pipeline file is written in: sources, steps, merges and stages.

A ``PipelineSpec`` is plain data: node classes with their settings objects, built node instances for the
steps that are made once, and the sources the app owns. ``PosePipeline`` turns it into callback chains at
startup. Every stage belongs to a clock domain: the frame domain runs on the camera's frame bang (the
source pushes, the async step emits on its thread), an output domain runs on the consumer's tick
(``PosePipeline.update``). The interpolators of a stage are the one crossing from the frame domain to an
output domain; a tick source joins at a stage the tick drives. A stage emits its poses and a group record
(``GroupRecord``): the per-pose steps and the group steps stamp the poses, the group steps write the record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Protocol, runtime_checkable

from ..features import BaseScalarFeature
from ..frame import FrameDict, FrameDictCallback
from ..nodes import FilterNode, GroupNode, InterpolatorNode
from ..window import WindowNodeSettings


class Domain(IntEnum):
    FRAME  = 0      # the camera's frame bang: the source, the inference step and the stages it feeds
    OUTPUT = 1      # the consumer's tick: the stages after the crossing


@runtime_checkable
class FrameSource(Protocol):
    """A source that pushes poses in the frame domain (``PosesFromTracklets``)."""

    @property
    def track_ids(self) -> list[int]:
        """The ids this source can emit."""
        ...

    def add_frames_callback(self, callback: FrameDictCallback) -> None: ...


@runtime_checkable
class TickSource(Protocol):
    """A source polled on the tick of an output domain (the dummy)."""

    @property
    def track_ids(self) -> list[int]:
        """The ids this source can emit."""
        ...

    @property
    def solo(self) -> bool:
        """True while this source's poses are to be the only ones past the merge."""
        ...

    def update(self, now: float) -> FrameDict:
        """The source's poses at ``now``, the tick's time (perf_counter)."""
        ...


@runtime_checkable
class AsyncNode(Protocol):
    """A step that emits later, on its own thread, and may drop frames (the crop and pose predictor)."""

    def submit(self, frames: FrameDict) -> None: ...

    def add_frames_callback(self, callback: FrameDictCallback) -> None: ...


class PoseStep:
    """A per-pose node class with its settings and fixed arguments: one instance per id that reaches the
    stage. The class is a ``FilterNode`` in a stage's steps, an ``InterpolatorNode`` in its interpolators."""

    __slots__ = ('node_cls', 'args')

    def __init__(self, node_cls: type[FilterNode] | type[InterpolatorNode], *args: Any) -> None:
        self.node_cls = node_cls
        self.args = args

    def build(self) -> Any:
        return self.node_cls(*self.args)

    def __repr__(self) -> str:
        return f"PoseStep({self.node_cls.__name__})"


class GroupStep:
    """A ``GroupNode`` instance, built once: every pose of the frame in, the same poses out, stamped, and the
    group record written."""

    __slots__ = ('node',)

    def __init__(self, node: GroupNode) -> None:
        self.node = node

    def __repr__(self) -> str:
        return f"GroupStep({type(self.node).__name__})"


class AsyncStep:
    """An ``AsyncNode`` instance, built once: the stage goes on from what the node emits, on its thread."""

    __slots__ = ('node',)

    def __init__(self, node: AsyncNode) -> None:
        self.node = node

    def __repr__(self) -> str:
        return f"AsyncStep({type(self.node).__name__})"


Step = PoseStep | GroupStep | AsyncStep


class Merge:
    """The inputs of an output-domain stage joined: the frame-domain stage its interpolators cross from and
    tick sources. The union of their poses, or those of the sources whose ``solo`` is on."""

    __slots__ = ('inputs',)

    def __init__(self, *inputs: int | TickSource) -> None:
        self.inputs = inputs

    def __repr__(self) -> str:
        return f"Merge{self.inputs}"


StageInput = int | FrameSource | TickSource | Merge


@dataclass(frozen=True)
class StageSpec:
    """One stage: its input (a stage, a source or a merge), its steps in order, its clock domain, the
    interpolators of the crossing (output domain only), its window settings and extra window features."""
    stage:           int
    input:           StageInput
    steps:           list[Step]
    domain:          Domain = Domain.FRAME
    interpolators:   list[PoseStep] = field(default_factory=list)
    window:          WindowNodeSettings | None = None
    window_features: list[type[BaseScalarFeature]] | None = None


@dataclass(frozen=True)
class PipelineSpec:
    """The stages in order; a stage's input is declared before it."""
    stages: list[StageSpec]
