# Standard library imports
import logging
from dataclasses import replace
from threading import Lock

# Local application imports
from .. import Tracklet, TrackingStatus

logger = logging.getLogger(__name__)


TrackletKey = tuple[int, int]  # (cam_id, external_id)


class TrackletIdPool:
    def __init__(self, max_size: int) -> None:
        self._available: set[int] = set(range(max_size))
        self._lock = Lock()

    def acquire(self) -> int:
        with self._lock:
            if not self._available:
                raise Exception("No more IDs available")
            min_id: int = min(self._available)
            self._available.remove(min_id)
            return min_id

    def release(self, obj: int) -> None:
        with self._lock:
            if obj in self._available:
                raise Exception(
                    f"ID {obj} is not currently in use and cannot be "
                    f"released. in use: {self._available}"
                )
            self._available.add(obj)

    def size(self) -> int:
        return len(self._available)

    def is_available(self, obj: int) -> bool:
        with self._lock:
            return obj in self._available

    @property
    def available(self) -> list[int]:
        with self._lock:
            return sorted(self._available)


class TrackletStore:
    """
    Stores per-camera tracklet observations and groups them into world identities.

    Observations are immutable per (cam_id, external_id) and never rewritten by
    cross-camera fusion: each observation's `cam_id`, `external_id`, `roi`, and
    `annotation` always reflect the camera it was actually seen by.

    World identities (ids drawn from the pool) own one or more observations. A
    person visible in two cameras simultaneously has two observations linked to
    the same world id; one is selected as "primary" by the tracker's view policy.
    Crossing a seam is handled by linking the new camera's observation into the
    existing world id — no merge, no rewrite, no flicker.
    """

    def __init__(self, max_players: int) -> None:
        self._tracklets: dict[TrackletKey, Tracklet] = {}
        self._world_members: dict[int, set[TrackletKey]] = {}
        self._world_for: dict[TrackletKey, int] = {}
        self._id_pool = TrackletIdPool(max_players)

    def __contains__(self, tracklet: Tracklet) -> bool:
        return self.get_world_id(tracklet.cam_id, tracklet.external_id) is not None

    def add_tracklet(self, tracklet: Tracklet, world_id: int | None = None) -> int | None:
        """
        Store a brand-new observation. If `world_id` is None, a fresh world id
        is acquired from the pool; otherwise the observation is linked into the
        existing world. Returns the world id, or None if the pool is exhausted.
        """
        key: TrackletKey = (tracklet.cam_id, tracklet.external_id)
        if key in self._tracklets:
            logger.warning(f"TrackletManager: add_tracklet on existing key {key}; use replace_tracklet.")
            return self._world_for.get(key)

        if world_id is None:
            try:
                world_id = self._id_pool.acquire()
            except Exception as e:
                logger.info(f"TrackletManager: No more world IDs available: {e}")
                return None
        elif world_id not in self._world_members:
            logger.warning(f"TrackletManager: add_tracklet linking to unknown world {world_id}.")
            return None

        self._tracklets[key] = replace(tracklet, id=world_id, status=TrackingStatus.NEW)
        self._world_members.setdefault(world_id, set()).add(key)
        self._world_for[key] = world_id
        return world_id

    def replace_tracklet(self, new_tracklet: Tracklet) -> int:
        """
        Refresh an existing observation, keyed by (cam_id, external_id).
        Preserves `created_at`. On LOST, `last_active` keeps the maximum to
        avoid rolling time backward. The observation's identity (cam_id /
        external_id) is the key and is never changed.
        """
        key: TrackletKey = (new_tracklet.cam_id, new_tracklet.external_id)
        old_tracklet: Tracklet | None = self._tracklets.get(key)
        if old_tracklet is None:
            logger.warning(f"TrackletManager: Attempted to replace non-existent tracklet {key}.")
            return -1

        status: TrackingStatus = new_tracklet.status
        if status == TrackingStatus.NEW:
            status = TrackingStatus.TRACKED  # a replaced tracklet can not be NEW

        last_active: float = new_tracklet.last_active
        if new_tracklet.status == TrackingStatus.LOST:
            last_active = max(old_tracklet.last_active, new_tracklet.last_active)

        world_id: int = self._world_for[key]
        self._tracklets[key] = replace(
            new_tracklet,
            id=world_id,
            created_at=old_tracklet.created_at,
            last_active=last_active,
            status=status,
        )
        return world_id

    def lose_tracklet(self, cam_id: int, external_id: int) -> None:
        key: TrackletKey = (cam_id, external_id)
        tracklet: Tracklet | None = self._tracklets.get(key)
        if tracklet is None:
            logger.warning(f"TrackletManager: Attempted to lose non-existent tracklet {key}.")
            return
        self._tracklets[key] = replace(tracklet, status=TrackingStatus.LOST)

    def retire_tracklet(self, cam_id: int, external_id: int) -> None:
        key: TrackletKey = (cam_id, external_id)
        tracklet: Tracklet | None = self._tracklets.get(key)
        if tracklet is None:
            logger.warning(f"TrackletManager: Attempted to retire non-existent tracklet {key}.")
            return
        self._tracklets[key] = replace(tracklet, status=TrackingStatus.REMOVED)

    def remove_tracklet(self, cam_id: int, external_id: int) -> None:
        """Delete an observation. Releases its world id if it was the last member."""
        key: TrackletKey = (cam_id, external_id)
        if self._tracklets.pop(key, None) is None:
            logger.warning(f"TrackletManager: Attempted to remove non-existent tracklet {key}.")
            return
        world_id: int | None = self._world_for.pop(key, None)
        if world_id is None:
            return
        members: set[TrackletKey] = self._world_members.get(world_id, set())
        members.discard(key)
        if not members:
            self._world_members.pop(world_id, None)
            self._id_pool.release(world_id)

    def merge_worlds(self, keep_id: int, drop_id: int) -> bool:
        """Move all observations from `drop_id` into `keep_id` and release `drop_id`."""
        if keep_id == drop_id:
            return False
        if keep_id not in self._world_members or drop_id not in self._world_members:
            logger.warning(f"TrackletManager: merge_worlds with unknown world id (keep={keep_id}, drop={drop_id}).")
            return False
        for key in self._world_members[drop_id]:
            self._world_for[key] = keep_id
            t: Tracklet = self._tracklets[key]
            self._tracklets[key] = replace(t, id=keep_id)
            self._world_members[keep_id].add(key)
        del self._world_members[drop_id]
        self._id_pool.release(drop_id)
        return True

    def get_world_id(self, cam_id: int, external_id: int) -> int | None:
        return self._world_for.get((cam_id, external_id))

    def get_tracklets(self, world_id: int) -> list[Tracklet]:
        keys: set[TrackletKey] = self._world_members.get(world_id, set())
        return [self._tracklets[k] for k in keys if k in self._tracklets]

    def all_tracklets(self) -> list[Tracklet]:
        return list(self._tracklets.values())

    def all_world_ids(self) -> list[int]:
        return list(self._world_members.keys())
