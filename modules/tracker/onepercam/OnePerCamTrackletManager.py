# Standard library imports
from dataclasses import replace
from threading import Lock
from typing import Optional

from pandas import Timestamp

# Local application imports
from modules.tracker.Tracklet import Tracklet, TrackingStatus

from modules.utils.HotReloadMethods import HotReloadMethods

class OnePerCamTrackletManager:
    def __init__(self, max_players: int) -> None:
        self._lock = Lock()
        self._max_size: int = max_players
        self._tracklets: dict[int, Tracklet] = {}

        hot_reload = HotReloadMethods(self.__class__, True)

    def __contains__(self, tracklet: Tracklet) -> bool:
        return self.get_id_by_cam_and_external_id(tracklet.cam_id, tracklet.external_id) is not None

    def add_tracklet(self, tracklet: Tracklet) -> Optional[int]:
        with self._lock:
            id: int = tracklet.cam_id

            if id in self._tracklets:
                print(f"ScreenBoundTrackletManager: Tracklet with cam_id {id} already exists. Skipping addition.")
                return None

            if id >= self._max_size:
                print(f"ScreenBoundTrackletManager: Tracklet with cam_id {id} exceeds max size {self._max_size}. Skipping addition.")
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
            if tracklet is None:
                print(f"TrackletManager: Attempted to remove non-existent tracklet with ID {id}.")

    def get_tracklet(self, id: int) -> Optional[Tracklet]:
        with self._lock:
            return self._tracklets.get(id)

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
                print(f"TrackletManager: Attempted to replace non-existent tracklet with ID {id}.")
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

    def lose_tracklet(self, id: int) -> None:
        with self._lock:
            tracklet: Tracklet | None = self._tracklets.get(id)
            if tracklet is None:
                print(f"TrackletManager: Attempted to retire non-existent tracklet with ID {id}.")
                return

            removed_tracklet: Tracklet = replace(
                tracklet,
                status=TrackingStatus.LOST,
                needs_notification=True
            )
            self._tracklets[id] = removed_tracklet

    def mark_all_as_notified(self) -> None:
        with self._lock:
            for id, tracklet in self._tracklets.items():
                if tracklet.needs_notification:
                    notified_tracklet: Tracklet = replace(
                        tracklet,
                        needs_notification=False
                    )
                    self._tracklets[id] = notified_tracklet

    def set_metadata(self, id: int, metadata) -> None:
        with self._lock:
            tracklet: Tracklet | None = self._tracklets.get(id)
            if tracklet is None:
                print(f"TrackletManager: Attempted to update metadata for non-existent tracklet with ID {id}.")
                return

            updated_tracklet: Tracklet = replace(
                tracklet,
                metadata=metadata
            )
            self._tracklets[id] = updated_tracklet