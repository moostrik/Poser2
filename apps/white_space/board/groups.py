from __future__ import annotations

from threading import Lock
from typing import Protocol

from modules.pose import GroupRecord


class HasGroups(Protocol):
    """Staged group record access."""
    def get_group(self, stage: int) -> GroupRecord: ...
    def set_group(self, stage: int, group: GroupRecord) -> None: ...


class GroupStoreMixin:
    """Thread-safe staged group record storage; a stage never written reads an empty record."""

    def __init__(self) -> None:
        self._group_lock = Lock()
        self._groups: dict[int, GroupRecord] = {}

    def get_group(self, stage: int) -> GroupRecord:
        with self._group_lock:
            return self._groups.get(stage, _EMPTY)

    def set_group(self, stage: int, group: GroupRecord) -> None:
        with self._group_lock:
            self._groups[stage] = group


_EMPTY = GroupRecord()
