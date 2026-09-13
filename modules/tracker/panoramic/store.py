# Standard library imports
import logging
from collections import deque
from dataclasses import replace
from threading import Lock

# Local application imports
from .. import Tracklet, TrackingStatus

logger = logging.getLogger(__name__)


ObsId = int
DeviceKey = tuple[int, int]  # (cam_id, external_id) — only meaningful while a device track lives


class TrackletIdPool:
    """FIFO id pool: released ids go to the back of the queue, so a freed id
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

    @property
    def available(self) -> list[int]:
        """Free ids in reuse order (front of the queue first)."""
        with self._lock:
            return list(self._queue)


class TrackletStore:
    """
    Stores per-camera tracklet observations and groups them into world identities.

    Observations are **host-owned**, keyed by an ``obs_id`` this store hands out and never
    reuses. A device tracklet id is only a token for the life of one device track: once the
    device reports ``REMOVED`` that number is dead, and the next tracklet carrying it is a
    different person. So ``(cam_id, external_id)`` is kept only in a *live* index, dropped the
    moment the device track ends (``end_device_track``). A reused device number then lands on a
    new observation, while the old one lives on as a ``LOST`` anchor until the tracker times it
    out — which is what lets a far camera still link across a seam after the near one gave up.
    Without that split, a newcomer inheriting a departed person's device id would be refreshed
    into the departed person's world.

    Observations are immutable and never rewritten by cross-camera fusion: each one's
    ``cam_id``, ``external_id``, ``roi`` and ``annotation`` always reflect the camera it was
    actually seen by.

    World identities (ids drawn from the pool) own one or more observations. A person visible in
    two cameras simultaneously has two observations linked to the same world id; one is selected
    as "primary" by the tracker's view policy. Crossing a seam is handled by linking the new
    camera's observation into the existing world id — no merge, no rewrite, no flicker.
    """

    def __init__(self, max_players: int) -> None:
        self._obs: dict[ObsId, Tracklet] = {}
        self._live: dict[DeviceKey, ObsId] = {}
        self._world_members: dict[int, set[ObsId]] = {}
        self._world_for: dict[ObsId, int] = {}
        self._next_obs_id: ObsId = 0
        self._id_pool = TrackletIdPool(max_players)

    def __contains__(self, tracklet: Tracklet) -> bool:
        return self.get_world_id(tracklet.cam_id, tracklet.external_id) is not None

    # ── live-index lookups ─────────────────────────────────────────────

    def get_world_id(self, cam_id: int, external_id: int) -> int | None:
        """The world of the LIVE device track with this id, or None. A device track that has
        ended is deliberately invisible here, even while its observation still anchors."""
        obs_id: ObsId | None = self._live.get((cam_id, external_id))
        return None if obs_id is None else self._world_for.get(obs_id)

    def get_live_observation(self, cam_id: int, external_id: int) -> Tracklet | None:
        obs_id: ObsId | None = self._live.get((cam_id, external_id))
        return None if obs_id is None else self._obs.get(obs_id)

    # ── mutation ───────────────────────────────────────────────────────

    def add_tracklet(self, tracklet: Tracklet, world_id: int | None = None) -> int | None:
        """
        Store a brand-new observation. If `world_id` is None, a fresh world id
        is acquired from the pool; otherwise the observation is linked into the
        existing world. Returns the world id, or None if the pool is exhausted.
        """
        key: DeviceKey = (tracklet.cam_id, tracklet.external_id)
        if key in self._live:
            logger.warning(f"add_tracklet on live key {key}; use replace_tracklet.")
            return self._world_for.get(self._live[key])

        if world_id is None:
            try:
                world_id = self._id_pool.acquire()
            except Exception as e:
                logger.info(f"No more world IDs available: {e}")
                return None
        elif world_id not in self._world_members:
            logger.warning(f"add_tracklet linking to unknown world {world_id}.")
            return None

        obs_id: ObsId = self._next_obs_id
        self._next_obs_id += 1

        self._obs[obs_id] = replace(tracklet, id=world_id, obs_id=obs_id, status=TrackingStatus.NEW)
        self._live[key] = obs_id
        self._world_members.setdefault(world_id, set()).add(obs_id)
        self._world_for[obs_id] = world_id
        return world_id

    def replace_tracklet(self, new_tracklet: Tracklet) -> int:
        """
        Refresh the live observation for this device track. Preserves `created_at` and the
        observation's identity. On LOST, `last_active` keeps the maximum to avoid rolling time
        backward.
        """
        key: DeviceKey = (new_tracklet.cam_id, new_tracklet.external_id)
        obs_id: ObsId | None = self._live.get(key)
        old_tracklet: Tracklet | None = None if obs_id is None else self._obs.get(obs_id)
        if obs_id is None or old_tracklet is None:
            logger.warning(f"Attempted to replace non-live tracklet {key}.")
            return -1

        status: TrackingStatus = new_tracklet.status
        if status == TrackingStatus.NEW:
            status = TrackingStatus.TRACKED  # a replaced tracklet can not be NEW

        last_active: float = new_tracklet.last_active
        if new_tracklet.status == TrackingStatus.LOST:
            last_active = max(old_tracklet.last_active, new_tracklet.last_active)

        world_id: int = self._world_for[obs_id]
        self._obs[obs_id] = replace(
            new_tracklet,
            id=world_id,
            obs_id=obs_id,
            created_at=old_tracklet.created_at,
            last_active=last_active,
            status=status,
        )
        return world_id

    def lose_tracklet(self, cam_id: int, external_id: int, latest: Tracklet | None = None) -> None:
        """The device has no detection this frame but still holds the track: mark LOST and keep
        it live, because the same device id legitimately comes back.

        `latest` is for a person the camera *does* see but the tracker does not count — beyond the
        zone's far edge. Their newest `roi` and `annotation` are kept, so the observation (and the
        panorama's mark) follows them walking out, but `last_active` is **not** advanced: that is
        the clock `emit_timeout` and `lost_timeout` run on, and restarting it would keep them
        forever. (`replace_tracklet` with a LOST copy cannot do this — it takes the newer time.)
        """
        obs_id: ObsId | None = self._live.get((cam_id, external_id))
        if obs_id is None or obs_id not in self._obs:
            logger.warning(f"Attempted to lose non-live tracklet {(cam_id, external_id)}.")
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
            logger.warning(f"Attempted to end non-live tracklet {key}.")
            return
        self._obs[obs_id] = replace(self._obs[obs_id], status=TrackingStatus.LOST)

    def retire_tracklet(self, obs_id: ObsId) -> None:
        """Timed out: mark REMOVED so the next tick deletes it."""
        if obs_id not in self._obs:
            logger.warning(f"Attempted to retire non-existent observation {obs_id}.")
            return
        self._obs[obs_id] = replace(self._obs[obs_id], status=TrackingStatus.REMOVED)

    def remove_tracklet(self, obs_id: ObsId) -> None:
        """Delete an observation. Releases its world id if it was the last member."""
        tracklet: Tracklet | None = self._obs.pop(obs_id, None)
        if tracklet is None:
            logger.warning(f"Attempted to remove non-existent observation {obs_id}.")
            return
        self._live.pop((tracklet.cam_id, tracklet.external_id), None)
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

    def get_tracklets(self, world_id: int) -> list[Tracklet]:
        obs_ids: set[ObsId] = self._world_members.get(world_id, set())
        return [self._obs[o] for o in obs_ids if o in self._obs]

    def all_tracklets(self) -> list[Tracklet]:
        return list(self._obs.values())

    def all_world_ids(self) -> list[int]:
        return list(self._world_members.keys())
