# Standard library imports
from threading import Lock
from typing import Optional

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
            tracklet.id = id
            tracklet.status = TrackingStatus.NEW
            self._tracklets[id] = tracklet
            return id

    def get_tracklet(self, id: int) -> Optional[Tracklet]:
        with self._lock:
            return self._tracklets.get(id, None)

    def set_tracklet(self, tracklet: Tracklet) -> None:
        with self._lock:
            if tracklet.id in self._tracklets:
                self._tracklets[tracklet.id] = tracklet
            else:
                print(f"PersonManager: Attempted to set non-existent tracklet with ID {tracklet.id}. Adding as new tracklet.")

    def remove_tracklet(self, id: int) -> None:
        with self._lock:
            tracklet: Tracklet | None = self._tracklets.pop(id, None)
            # tracklet.status = TrackingStatus.REMOVED
            if tracklet is not None:
                self._id_pool.release(id)
            else:
                print(f"PersonManager: Attempted to remove non-existent tracklet with ID {id}.")

    def all_tracklets(self) -> list[Tracklet]:
        with self._lock:
            return list(self._tracklets.values())

    def get_tracklet_by_cam_and_external_id(self, cam_id: int, external_id: int) -> Optional[Tracklet]:
        with self._lock:
            for tracklet in self._tracklets.values():
                if tracklet.cam_id == cam_id and tracklet.external_id == external_id:
                    return tracklet
            return None

    def replace_tracklet(self, old_tracklet: Tracklet, new_tracklet: Tracklet) -> None:
        """
        Replace an existing tracklet in the manager with a new tracklet object.

        The new tracklet will:
        - Take the id and start_time from the old tracklet.
        - Have its status set to TRACKED if its current status is NEW.
        - Replace the old tracklet in the manager's dictionary.

        Args:
            old_tracklet (Person): The tracklet currently in the manager to be replaced.
            new_tracklet (Person): The new tracklet object to insert, inheriting id and start_time from old_tracklet.
        """
        if self.get_tracklet(old_tracklet.id) is None:
            print(f"PersonManager: Attempted to replace non-existent tracklet with ID {old_tracklet.id}.")
            return

        with self._lock:
            # Transfer id and start_time
            new_tracklet.id = old_tracklet.id
            new_tracklet.created_at = old_tracklet.created_at

            # Update status if needed
            if new_tracklet.status == TrackingStatus.NEW:
                new_tracklet.status = TrackingStatus.TRACKED

            if new_tracklet.status == TrackingStatus.LOST:
                new_tracklet.last_seen = old_tracklet.last_seen

            if new_tracklet.status == TrackingStatus.REMOVED:
                print(f"PersonManager: Attempted to replace tracklet with ID {new_tracklet.id} with status REMOVED. This should not happen.")
                # return

            # Replace in the dict
            self._tracklets[old_tracklet.id] = new_tracklet

    def merge_tracklets(self, keep: Tracklet, remove: Tracklet) -> int:
        """
        Merge two Person objects into a single entry in the manager.

        The resulting tracklet will:
        - Use the id and start_time of the older tracklet (the one with the earlier start_time).
        - Use all other attributes from the 'keep' tracklet.
        - Remove both original tracklets from the manager and release the ID of the newer tracklet.
        - Add the merged tracklet back to the manager with the merged id.

        Args:
            keep (Person): The tracklet whose data (except id and start_time) will be kept.
            remove (Person): The tracklet whose id and start_time may be used if older.

        Returns:
            int: The id of the tracklet that was removed and released, or -1 if the merge was not successful.
        """

        with self._lock:
            # Check for invalid IDs
            if keep.id in (-1, None):
                print(f"PersonManager: Cannot merge tracklets with uninitialized id (keep.id={keep.id}, remove.id={remove.id})")
                return -1

            if keep.id == remove.id:
                print(f"PersonManager: Attempted to merge the same tracklet {keep.id}.")
                return -1

            # Determine which tracklet is oldest
            if keep.age_in_seconds >= remove.age_in_seconds:
                oldest, newest = keep, remove
            else:
                oldest, newest = remove, keep

            # Save the id and start_time of the oldest
            merged_id: int = oldest.id
            other_id: int = newest.id

            # Use all other data from the newest (the 'keep' tracklet)
            merged_tracklet = Tracklet(
                id=merged_id,
                cam_id=keep.cam_id,
                created_at=oldest.created_at, # Use the created_at of the oldest tracklet
                last_seen=keep.last_seen,
                status=keep.status,
                roi=keep.roi,
                _external_tracklet=keep._external_tracklet,
                tracker_info=keep.tracker_info
            )

            if keep.status == TrackingStatus.NEW:
                merged_tracklet.status = TrackingStatus.TRACKED

            # Remove both old tracklets from the manager
            self._tracklets.pop(merged_id, None)
            if other_id != -1:
                self._tracklets.pop(other_id, None)
                if other_id != merged_id:
                    self._id_pool.release(other_id)

            # Add the merged tracklet with merged_id
            self._tracklets[merged_id] = merged_tracklet

            return other_id

