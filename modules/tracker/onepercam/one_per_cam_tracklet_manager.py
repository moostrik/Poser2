# Standard library imports
import logging
from dataclasses import replace
from threading import Lock

# Local application imports
from .. import Tracklet, TrackingStatus

logger = logging.getLogger(__name__)


class OnePerCamTrackletManager:
    def __init__(self, max_players: int) -> None:
        self._lock = Lock()
        self._max_size: int = max_players
        self._tracklets: dict[int, Tracklet] = {}

        # hot_reload = HotReloadMethods(self.__class__, True)

    def __contains__(self, tracklet: Tracklet) -> bool:
        return self.get_id_by_cam_and_external_id(tracklet.cam_id, tracklet.external_id) is not None

    def add_tracklet(self, tracklet: Tracklet) -> int | None:
        with self._lock:
            tracklet_id: int = tracklet.cam_id

            if tracklet_id in self._tracklets:
                logger.warning(
                    f"OnePerCamTrackletManager: Tracklet with cam_id "
                    f"{tracklet_id} already exists. Skipping addition."
                )
                return None

            if tracklet_id >= self._max_size:
                logger.warning(
                    f"OnePerCamTrackletManager: Tracklet with cam_id "
                    f"{tracklet_id} exceeds max size {self._max_size}. "
                    f"Skipping addition."
                )
                return None

            new_tracklet: Tracklet = replace(
                tracklet,
                id=tracklet_id,
                status=TrackingStatus.NEW,
            )
            self._tracklets[tracklet_id] = new_tracklet
            return tracklet_id

    def remove_tracklet(self, id: int) -> None:
        with self._lock:
            tracklet: Tracklet | None = self._tracklets.pop(id, None)
            if tracklet is None:
                logger.warning(f"TrackletManager: Attempted to remove non-existent tracklet with ID {id}.")

    def get_tracklet(self, id: int) -> Tracklet | None:
        with self._lock:
            return self._tracklets.get(id)

    def all_tracklets(self) -> list[Tracklet]:
        with self._lock:
            return list(self._tracklets.values())

    def get_id_by_cam_and_external_id(self, cam_id: int, external_id: int) -> int | None:
        with self._lock:
            for tracklet in self._tracklets.values():
                if tracklet.cam_id == cam_id and tracklet.external_id == external_id:
                    return tracklet.id
            return None

    def replace_tracklet(self, id: int, new_tracklet: Tracklet) -> int:
        with self._lock:
            old_tracklet: Tracklet | None = self._tracklets.get(id)
            if old_tracklet is None:
                logger.warning(f"TrackletManager: Attempted to replace non-existent tracklet with ID {id}.")
                return -1

            status: TrackingStatus = new_tracklet.status
            if status == TrackingStatus.NEW:
                status = TrackingStatus.TRACKED  # A replaced tracklet can not be NEW

            last_active: float = new_tracklet.last_active
            if new_tracklet.status == TrackingStatus.LOST:
                last_active = old_tracklet.last_active

            # Create a new instance with updated fields
            updated_tracklet: Tracklet = replace(
                new_tracklet,
                id=id,
                created_at=old_tracklet.created_at,
                last_active=last_active,
                status=status,
            )
            self._tracklets[id] = updated_tracklet
            return id

    def retire_tracklet(self, id: int) -> None:
        with self._lock:
            tracklet: Tracklet | None = self._tracklets.get(id)
            if tracklet is None:
                logger.warning(f"TrackletManager: Attempted to retire non-existent tracklet with ID {id}.")
                return

            removed_tracklet: Tracklet = replace(
                tracklet,
                status=TrackingStatus.REMOVED,
            )
            self._tracklets[id] = removed_tracklet

    def lose_tracklet(self, id: int) -> None:
        with self._lock:
            tracklet: Tracklet | None = self._tracklets.get(id)
            if tracklet is None:
                logger.warning(f"TrackletManager: Attempted to lose non-existent tracklet with ID {id}.")
                return

            lost_tracklet: Tracklet = replace(
                tracklet,
                status=TrackingStatus.LOST,
            )
            self._tracklets[id] = lost_tracklet