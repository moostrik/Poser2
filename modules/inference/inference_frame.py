"""An inference frame: one camera's moment, its image and what the models found in it, in one piece.

As the pose `Frame` carries its features, an inference frame carries its results keyed by their type: `frame[Poses]`
never raises, and a type the frame does not carry reads as its empty one (no players). Which types an app's frames
carry is the app's: `InferenceFrameSync` is made with them."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable, Self, TypeAlias, TypeVar, cast

from modules.pose import FrameDict

from .preprocessing import GpuDepth, GpuImage

if TYPE_CHECKING:
    from .segmentation import ImageDict


@dataclass(frozen=True)
class InferenceResult:
    """What one model, or the depth's upload, gives for one camera's image captured at `timestamp`; `synced` when it
    is that moment's own, not the newest from before it."""
    cam_id: int
    timestamp: float
    synced: bool = True

    @classmethod
    def empty(cls, cam_id: int, timestamp: float) -> Self:
        """The result of a moment without it: no players, not synced."""
        return cls(cam_id, timestamp, False)


@dataclass(frozen=True)
class Depth(InferenceResult):
    """The depth uploaded with the image; a frame of an app with depth is complete only with it."""
    depth: GpuDepth | None = None

    @classmethod
    def of(cls, depth: GpuDepth) -> Depth:
        return cls(depth.cam_id, depth.timestamp, True, depth)


@dataclass(frozen=True)
class Poses(InferenceResult):
    """The poses of the image's players, by track id."""
    poses: FrameDict = field(default_factory=dict)

    @classmethod
    def by_camera(cls, poses: FrameDict) -> list[Poses]:
        """A pipeline stage's poses as one result per camera, each stamped with its image's capture time."""
        by_cam: dict[int, FrameDict] = {}
        for track_id, pose in poses.items():
            by_cam.setdefault(pose.cam_id, {})[track_id] = pose
        return [cls(cam_id, max(pose.timestamp for pose in cam_poses.values()), True, cam_poses)
                for cam_id, cam_poses in by_cam.items()]


@dataclass(frozen=True)
class Masks(InferenceResult):
    """The segmentation's masks of the image's players, by track id."""
    masks: ImageDict = field(default_factory=dict)

    @classmethod
    def by_camera(cls, poses: FrameDict, masks: ImageDict) -> list[Masks]:
        """The segmentation's masks as one result per camera, each found by the pose it was cut for; a result
        without masks gives none."""
        by_cam: dict[int, ImageDict] = {}
        times: dict[int, float] = {}
        for track_id, mask in masks.items():
            pose = poses.get(track_id)
            if pose is None:
                continue
            by_cam.setdefault(pose.cam_id, {})[track_id] = mask
            times[pose.cam_id] = max(times.get(pose.cam_id, pose.timestamp), pose.timestamp)
        return [cls(cam_id, times[cam_id], True, cam_masks) for cam_id, cam_masks in by_cam.items()]


R = TypeVar('R', bound=InferenceResult)


class InferenceFrame:
    """One camera's moment: its image on the GPU, captured at `timestamp` (`CameraFrame.timestamp`), and its results
    by type."""

    __slots__ = ('_cam_id', '_timestamp', '_image', '_results')

    def __init__(self, image: GpuImage, results: Iterable[InferenceResult] = (), timestamp: float | None = None) -> None:
        object.__setattr__(self, '_cam_id', image.cam_id)
        object.__setattr__(self, '_timestamp', image.timestamp if timestamp is None else timestamp)
        object.__setattr__(self, '_image', image)
        object.__setattr__(self, '_results', {type(result): result for result in results})

    @property
    def cam_id(self) -> int:
        return self._cam_id

    @property
    def timestamp(self) -> float:
        return self._timestamp

    @property
    def image(self) -> GpuImage:
        return self._image

    def __getitem__(self, kind: type[R]) -> R:
        """The result of that type; its empty one when the frame does not carry it."""
        result: InferenceResult | None = self._results.get(kind)
        return cast(R, result) if result is not None else kind.empty(self._cam_id, self._timestamp)

    def __contains__(self, kind: type) -> bool:
        return kind in self._results

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("InferenceFrame is immutable")

    def __repr__(self) -> str:
        return f"InferenceFrame(cam={self._cam_id}, t={self._timestamp:.4f}, results=[{', '.join(t.__name__ for t in self._results)}])"


InferenceFrameCallback: TypeAlias = Callable[[InferenceFrame], None]
