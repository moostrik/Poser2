"""GroupRecord — the group's record a stage emits beside its poses.

Typed values of a frame that belong to everyone in it together and to no single pose: the group's similarity
level, a gate on it, the poses hit this tick. The same contract as ``Frame``: immutable, keyed by feature
class, ``group[FeatureType]`` never raises (a NaN dummy when the feature was never written), and the record
carries the stamp of the frames it was made with. A group step (``GroupNode.record``) writes it with
``replace_group``.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from ..features import BaseFeature

T = TypeVar('T')


class GroupRecord:
    """Immutable group record: a stamp and a feature dict; see the module docstring."""

    __slots__ = ('_timestamp', '_features')

    def __init__(self, timestamp: float = 0.0, features: dict[type[BaseFeature], BaseFeature] | None = None) -> None:
        object.__setattr__(self, '_timestamp', timestamp)
        object.__setattr__(self, '_features', dict(features) if features else {})

    @property
    def timestamp(self) -> float:
        """The stamp of the frames the record was made with: the tick in an output domain."""
        return self._timestamp

    def __getitem__(self, feature_type: type[T]) -> T:
        """The feature, or its NaN dummy when it was never written."""
        try:
            return self._features[feature_type]
        except KeyError:
            return feature_type.create_dummy()  # type: ignore[union-attr]

    def __contains__(self, feature_type: type) -> bool:
        return feature_type in self._features

    def __len__(self) -> int:
        return len(self._features)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("GroupRecord is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("GroupRecord is immutable")

    def __repr__(self) -> str:
        return f"GroupRecord(features=[{', '.join(t.__name__ for t in self._features)}])"


def replace_group(group: GroupRecord, updates: dict[type[BaseFeature], BaseFeature], timestamp: float | None = None) -> GroupRecord:
    """A new record with ``updates`` merged over the group's features; the group's stamp unless a new one is given."""
    merged = dict(group._features)
    merged.update(updates)
    return GroupRecord(timestamp=group._timestamp if timestamp is None else timestamp, features=merged)


GroupCallback = Callable[[GroupRecord], None]
