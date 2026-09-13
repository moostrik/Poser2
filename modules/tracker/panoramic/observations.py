# Standard library imports
import logging
from collections import deque
from dataclasses import replace
from threading import Lock

# Local application imports
from ..tracklet import Tracklet, TrackingStatus

logger = logging.getLogger(__name__)


ObsId = int
DeviceKey = tuple[int, int]  # (cam_id, external_id) — only meaningful while a device track lives


class WorldIdPool:
    """FIFO pool of world ids: released ids go to the back of the queue, so a freed id
    is reused as late as possible instead of being handed to the next arrival."""

    def __init__(self, max_size: int) -> None:
        self._queue: deque[int] = deque(range(max_size))
        self._available: set[int] = set(range(max_size))
        self._lock = Lock()

    def acquire(self) -> int:
        with self._lock:
            if not self._queue:
                raise Exception("No more IDs available")
            id_: int = self._queue.popleft()
            self._available.remove(id_)
            return id_

    def release(self, obj: int) -> None:
        with self._lock:
            if obj in self._available:
                raise Exception(
                    f"ID {obj} is not currently in use and cannot be "
                    f"released. available: {sorted(self._available)}"
                )
            self._queue.append(obj)
            self._available.add(obj)

    def size(self) -> int:
        return len(self._queue)

    def is_available(self, obj: int) -> bool:
        with self._lock:
            return obj in self._available


class ObservationStore:
    """
    Stores per-camera observations and groups them into world identities.

    Observations are **host-owned**, keyed by an ``obs_id`` this store hands out and never
    reuses. A device tracklet id is only a token for the life of one device track: once the
    device reports ``REMOVED`` that number is dead, and the next tracklet carrying it is a
    different person. So ``(cam_id, external_id)`` is kept only in a *live* index, dropped the
    moment the device track ends (``end_device_track``). A reused device number then lands on a
    new observation, while the old one lives on as a ``LOST`` anchor until the tracker times it
    out — which is what lets a far camera still link across a seam after the near one gave up.

    Observations are immutable and never rewritten when worlds are joined: each one's ``cam_id``,
    ``external_id``, ``roi`` and ``annotation`` always reflect the camera it was actually seen by.

    World identities (ids drawn from the pool) own one or more observations. A person visible in
    two cameras simultaneously has two observations linked to the same world id; one is selected
    as primary (`Seams.pick_primary`).
    """

    def __init__(self, max_players: int) -> None:
        self._obs: dict[ObsId, Tracklet] = {}
        self._live: dict[DeviceKey, ObsId] = {}
        self._world_members: dict[int, set[ObsId]] = {}
        self._world_for: dict[ObsId, int] = {}
        self._next_obs_id: ObsId = 0
        self._id_pool = WorldIdPool(max_players)

    # ── live-index lookups ─────────────────────────────────────────────

    def world_of(self, cam_id: int, external_id: int) -> int | None:
        """The world of the LIVE device track with this id, or None. A device track that has
        ended is deliberately invisible here, even while its observation still anchors."""
        obs_id: ObsId | None = self._live.get((cam_id, external_id))
        return None if obs_id is None else self._world_for.get(obs_id)

    def live(self, cam_id: int, external_id: int) -> Tracklet | None:
        """The observation of the LIVE device track with this id, or None."""
        obs_id: ObsId | None = self._live.get((cam_id, external_id))
        return None if obs_id is None else self._obs.get(obs_id)

    # ── mutation ───────────────────────────────────────────────────────

    def add(self, tracklet: Tracklet, world_id: int | None = None) -> int | None:
        """
        Store a brand-new observation. If `world_id` is None, a fresh world id
        is acquired from the pool; otherwise the observation is linked into the
        existing world. Returns the world id, or None if the pool is exhausted.
        """
        key: DeviceKey = (tracklet.cam_id, tracklet.external_id)
        if key in self._live:
            logger.warning(f"add on live key {key}; use refresh.")
            return self._world_for.get(self._live[key])

        if world_id is None:
            try:
                world_id = self._id_pool.acquire()
            except Exception as e:
                logger.info(f"No more world IDs available: {e}")
                return None
        elif world_id not in self._world_members:
            logger.warning(f"add linking to unknown world {world_id}.")
            return None

        obs_id: ObsId = self._next_obs_id
        self._next_obs_id += 1

        self._obs[obs_id] = replace(tracklet, id=world_id, obs_id=obs_id, status=TrackingStatus.NEW)
        self._live[key] = obs_id
        self._world_members.setdefault(world_id, set()).add(obs_id)
        self._world_for[obs_id] = world_id
        return world_id

    def refresh(self, new_tracklet: Tracklet) -> int:
        """
        Refresh the live observation for this device track with a new detection. Preserves
        `created_at` and the observation's identity. A missed detection is `lose`.
        """
        key: DeviceKey = (new_tracklet.cam_id, new_tracklet.external_id)
        obs_id: ObsId | None = self._live.get(key)
        old_tracklet: Tracklet | None = None if obs_id is None else self._obs.get(obs_id)
        if obs_id is None or old_tracklet is None:
            logger.warning(f"Attempted to refresh non-live observation {key}.")
            return -1

        status: TrackingStatus = new_tracklet.status
        if status == TrackingStatus.NEW:
            status = TrackingStatus.TRACKED  # a replaced tracklet can not be NEW

        world_id: int = self._world_for[obs_id]
        self._obs[obs_id] = replace(
            new_tracklet,
            id=world_id,
            obs_id=obs_id,
            created_at=old_tracklet.created_at,
            status=status,
        )
        return world_id

    def lose(self, cam_id: int, external_id: int, latest: Tracklet | None = None) -> None:
        """The device has no detection this frame but still holds the track: mark LOST and keep
        it live, because the same device id legitimately comes back.

        `latest` is for a person the camera *does* see but the tracker does not count — past the
        zone's far edge, or a box a filter rejects. Their newest `roi` and `annotation` are kept, so
        the observation (and the panorama's mark) follows them, but `last_active` is **not** advanced:
        that is the clock `lost_timeout` and pose's `detection_timeout` run on, and restarting it would
        keep them forever. (`refresh` with a LOST copy cannot do this — it takes the newer time.)
        """
        obs_id: ObsId | None = self._live.get((cam_id, external_id))
        if obs_id is None or obs_id not in self._obs:
            logger.warning(f"Attempted to lose non-live observation {(cam_id, external_id)}.")
            return
        old: Tracklet = self._obs[obs_id]
        if latest is None:
            self._obs[obs_id] = replace(old, status=TrackingStatus.LOST)
        else:
            self._obs[obs_id] = replace(old, status=TrackingStatus.LOST,
                                        roi=latest.roi, annotation=latest.annotation)

    def end_device_track(self, cam_id: int, external_id: int) -> None:
        """The device has dropped the track for good. The observation stays as a LOST anchor
        until the tracker times it out, but leaves the live index so the device is free to hand
        that id to someone else without the newcomer inheriting this world."""
        key: DeviceKey = (cam_id, external_id)
        obs_id: ObsId | None = self._live.pop(key, None)
        if obs_id is None or obs_id not in self._obs:
            logger.warning(f"Attempted to end non-live observation {key}.")
            return
        self._obs[obs_id] = replace(self._obs[obs_id], status=TrackingStatus.LOST)

    def retire(self, obs_id: ObsId) -> None:
        """Timed out: mark REMOVED so the next tick deletes it."""
        if obs_id not in self._obs:
            logger.warning(f"Attempted to retire non-existent observation {obs_id}.")
            return
        self._obs[obs_id] = replace(self._obs[obs_id], status=TrackingStatus.REMOVED)

    def remove(self, obs_id: ObsId) -> None:
        """Delete an observation. Releases its world id if it was the last member."""
        tracklet: Tracklet | None = self._obs.pop(obs_id, None)
        if tracklet is None:
            logger.warning(f"Attempted to remove non-existent observation {obs_id}.")
            return
        # Only if the key is still this observation's: after `end_device_track` the device may have
        # handed the same id to a newer observation, whose entry this must not take.
        key: DeviceKey = (tracklet.cam_id, tracklet.external_id)
        if self._live.get(key) == obs_id:
            del self._live[key]
        world_id: int | None = self._world_for.pop(obs_id, None)
        if world_id is None:
            return
        members: set[ObsId] = self._world_members.get(world_id, set())
        members.discard(obs_id)
        if not members:
            self._world_members.pop(world_id, None)
            self._id_pool.release(world_id)

    def merge_worlds(self, keep_id: int, drop_id: int) -> bool:
        """Move all observations from `drop_id` into `keep_id` and release `drop_id`."""
        if keep_id == drop_id:
            return False
        if keep_id not in self._world_members or drop_id not in self._world_members:
            logger.warning(f"merge_worlds with unknown world id (keep={keep_id}, drop={drop_id}).")
            return False
        for obs_id in self._world_members[drop_id]:
            self._world_for[obs_id] = keep_id
            self._obs[obs_id] = replace(self._obs[obs_id], id=keep_id)
            self._world_members[keep_id].add(obs_id)
        del self._world_members[drop_id]
        self._id_pool.release(drop_id)
        return True

    # ── reads ──────────────────────────────────────────────────────────

    def has_free_id(self) -> bool:
        """Whether a new world can be started."""
        return self._id_pool.size() > 0

    def camera_sees_world(self, cam_id: int, world_id: int) -> bool:
        """Whether this camera already actively tracks someone in this world.

        One camera never sees one person twice — the device de-duplicates — so a second id from it
        is a second person, and no rule may join the two. A merge would be sticky: nothing splits a
        world again."""
        return any(t.cam_id == cam_id and t.is_active for t in self.members(world_id))

    def members(self, world_id: int) -> list[Tracklet]:
        """Every observation in this world."""
        obs_ids: set[ObsId] = self._world_members.get(world_id, set())
        return [self._obs[o] for o in obs_ids if o in self._obs]

    def all(self) -> list[Tracklet]:
        """Every observation, in every world."""
        return list(self._obs.values())

    def world_ids(self) -> list[int]:
        return list(self._world_members.keys())
