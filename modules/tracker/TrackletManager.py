# Standard library imports
from dataclasses import replace
from threading import Lock
from typing import Optional

from pandas import Timestamp

# Local application imports
from modules.tracker.Tracklet import Tracklet, TrackingStatus

from modules.utils.HotReloadMethods import HotReloadMethods

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
                raise Exception(f"ID {obj} is not currently in use and cannot be released. in use: {self._available}")
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

class TrackletManager:
    def __init__(self, max_players: int) -> None:
        self._tracklets: dict[int, Tracklet] = {}
        self._id_pool = TrackletIdPool(max_players)
        self._lock = Lock()

        hot_reload = HotReloadMethods(self.__class__, True)

    def add_tracklet(self, tracklet: Tracklet) -> Optional[int]:
        with self._lock:
            try:
                id = self._id_pool.acquire()
            except Exception as e:
                print(f"PersonManager: No more IDs available: {e}")
                return None

            new_tracklet: Tracklet = replace(
                tracklet,
                id=id,
                status=TrackingStatus.NEW,
                needs_notification=True
            )
            self._tracklets[id] = new_tracklet
            return id

    def remove_tracklet(self, id: int) -> None:
        with self._lock:
            tracklet: Tracklet | None = self._tracklets.pop(id, None)
            if tracklet is not None:
                self._id_pool.release(id)
            else:
                print(f"PersonManager: Attempted to remove non-existent tracklet with ID {id}.")

    def all_tracklets(self) -> list[Tracklet]:
        with self._lock:
            return list(self._tracklets.values())

    def get_id_by_cam_and_external_id(self, cam_id: int, external_id: int) -> Optional[int]:
        with self._lock:
            for tracklet in self._tracklets.values():
                if tracklet.cam_id == cam_id and tracklet.external_id == external_id:
                    return tracklet.id
            return None

    def replace_tracklet(self, id: int, new_tracklet: Tracklet) -> int:
        with self._lock:
            old_tracklet: Tracklet | None = self._tracklets.get(id)
            if old_tracklet is None:
                print(f"PersonManager: Attempted to replace non-existent tracklet with ID {id}.")
                return -1

            status: TrackingStatus = new_tracklet.status
            if status == TrackingStatus.NEW:
                status = TrackingStatus.TRACKED  # A replaced tracklet can not be NEW

            last_active: Timestamp = new_tracklet.last_active
            if new_tracklet.status == TrackingStatus.LOST:
                last_active = old_tracklet.last_active

            # Create a new instance with updated fields
            updated_tracklet: Tracklet = replace(
                new_tracklet,
                id=id,
                created_at=old_tracklet.created_at,
                last_active=last_active,
                status=status,
                needs_notification=True
            )
            self._tracklets[id] = updated_tracklet
            return id

    def merge_tracklets(self, keep_id: int, remove_id: int) -> tuple[int, int]:
        with self._lock:
            # Validate IDs
            if keep_id in (-1, None) or keep_id == remove_id:
                print(f"TrackletManager: Invalid merge (keep.id={keep_id}, remove.id={remove_id})")
                return -1, -1

            keep: Optional[Tracklet] = self._tracklets.get(keep_id)
            remove: Optional[Tracklet] = self._tracklets.get(remove_id)

            if keep is None or remove is None:
                print(f"TrackletManager: One of the tracklets in the merge {keep_id} and {remove_id} is None. Skipping merge.")
                return -1, -1

            if not keep.is_active:
                print(f"TrackletManager: Cannot merge tracklet with status {keep.status} (keep.id={keep.id}, remove.id={remove.id})")
                return -1, -1


            # Determine which tracklet is oldest
            if keep.age_in_seconds >= remove.age_in_seconds:
                merge, other = keep, remove
            else:
                merge, other = remove, keep

            # Use all other data from the newest (the 'keep' tracklet)
            merged_tracklet: Tracklet = replace(
                keep,
                id=merge.id,
                created_at=merge.created_at,
                status=TrackingStatus.TRACKED,
                needs_notification=True
            )
            self._tracklets[merge.id] = merged_tracklet

            needs_notification: bool = True
            if other.status == TrackingStatus.NEW: # if a tracklet is just added and instantly merged, we don't want to mark it as updated
                needs_notification = False

            other_tracklet: Tracklet = replace(
                remove,
                id=other.id,
                created_at=other.created_at,
                status=TrackingStatus.REMOVED,
                needs_notification=needs_notification
            )
            self._tracklets[other.id] = other_tracklet

            return merge.id, other.id

    def retire_tracklet(self, id: int) -> None:
        with self._lock:
            tracklet: Tracklet | None = self._tracklets.get(id)
            if tracklet is None:
                print(f"TrackletManager: Attempted to retire non-existent tracklet with ID {id}.")
                return

            removed_tracklet: Tracklet = replace(
                tracklet,
                status=TrackingStatus.REMOVED,
                needs_notification=True
            )
            self._tracklets[id] = removed_tracklet

    def mark_all_as_notified(self) -> None:
        with self._lock:
            for id, tracklet in self._tracklets.items():
                if tracklet.needs_notification:
                    updated_tracklet: Tracklet = replace(
                        tracklet,
                        needs_notification=False
                    )
                    self._tracklets[id] = updated_tracklet

