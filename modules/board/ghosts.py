from __future__ import annotations

from threading import Lock
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.pose.frame import Frame


class HasGhosts(Protocol):
    """Read/write access to the published ghost snapshot (frozen virtual persons)."""
    def get_ghosts(self) -> dict[int, Frame]: ...
    def set_ghosts(self, ghosts: dict[int, Frame]) -> None: ...


class GhostStoreMixin:
    """Thread-safe store for the latest ghost snapshot.

    Holds a published view of a producer's ghost registry (refreshed each tick), keyed by the
    ghost's ``track_id``. Read-only for consumers — not the source of truth.
    """

    def __init__(self) -> None:
        self._ghost_lock = Lock()
        self._ghosts: dict[int, Frame] = {}

    def get_ghosts(self) -> dict[int, Frame]:
        with self._ghost_lock:
            return dict(self._ghosts)

    def set_ghosts(self, ghosts: dict[int, Frame]) -> None:
        with self._ghost_lock:
            self._ghosts = dict(ghosts)
