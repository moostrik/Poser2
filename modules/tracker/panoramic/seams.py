# Standard library imports
import math
from itertools import combinations

# Local application imports
from ..tracklet import Tracklet
from .annotation import Annotation
from .observations import ObservationStore
from .rig import Rig, height_is_measured
from .settings import TrackerSettings


class Seams:
    """Where two cameras' fields meet: the rules of `SeamSettings`.

    Which world a new observation inside the overlap links to (`linked_world`), which worlds
    turn out to be one person seen twice (`collapse_worlds`), and which camera's view a world is
    emitted as, handing over between cameras without flicker (`pick_primary`). The dead zone is the
    tracker intake's. Owns the handover state, and runs on the tracker thread only.
    """

    def __init__(self, store: ObservationStore, rig: Rig, config: TrackerSettings) -> None:
        self._store: ObservationStore = store
        self._rig: Rig = rig
        self._config: TrackerSettings = config
        # Last emitted primary per world id, as an observation id — for hysteresis — and when a
        # LOST primary was last replaced, for `seam.hold`.
        self._primary_for_world: dict[int, int] = {}
        self._handed_over_at: dict[int, float] = {}

    # ── linking ────────────────────────────────────────────────────────

    def linked_world(self, new_tracklet: Tracklet) -> int | None:
        """The world whose other-camera observation matches ``new_tracklet`` (closest in azimuth
        wins), or None. LOST observations still anchor, so the link survives the near camera losing
        the person first."""
        assert isinstance(new_tracklet.annotation, Annotation)
        best_world: int | None = None
        best_diff: float = float('inf')
        for t in self._store.all_tracklets():
            if t.is_removed:
                continue
            if not self._observations_match(new_tracklet, t):
                continue
            if self._store.camera_sees_world(new_tracklet.cam_id, t.id):
                continue
            assert isinstance(t.annotation, Annotation)
            diff: float = self._rig.angle_diff(new_tracklet.annotation.world_angle, t.annotation.world_angle)
            if diff < best_diff:
                best_diff = diff
                best_world = t.id
        return best_world

    def collapse_worlds(self) -> None:
        """Late safety net: merge worlds whose observations match each other (ambiguous
        simultaneous arrivals that each got their own world). Older world wins."""
        for keep_id, drop_id in self._collapse_pairs():
            if self._store.merge_worlds(keep_id, drop_id):
                self._primary_for_world.pop(drop_id, None)
                self._handed_over_at.pop(drop_id, None)

    # ── handover ───────────────────────────────────────────────────────

    def pick_primary(self, world_id: int, now: float) -> Tracklet | None:
        """The one view a world is emitted as: sticky, but never a LOST view while another is active.

        - **No member active**: the most recently seen one.
        - **The primary lost the person**: the best-placed active view takes over at once. The show
          reads `is_active`, so holding a LOST primary would drop a person a camera still sees.
        - **Otherwise** the primary yields only to a view whose distance from its field edge beats
          its own by the ``seam.hysteresis`` ratio.

        The two guards cover different switches. The ratio keeps an active-to-active handover from
        bouncing, since going back needs the ratio again. A forced handover skipped the ratio, so the
        new primary is often the worse-placed view and the ratio would hand the person straight back
        the moment the old camera returns; ``seam.hold`` blocks that for a while after a forced
        handover, so a one-frame miss costs one camera switch, not two.
        """
        members: list[Tracklet] = [t for t in self._store.get_tracklets(world_id)
                                   if not t.is_removed and isinstance(t.annotation, Annotation)]
        if not members:
            return None

        def edge(t: Tracklet) -> float:
            assert isinstance(t.annotation, Annotation)
            return self._rig.angle_from_edge(t.annotation.local_angle)

        current_key: int | None = self._primary_for_world.get(world_id)
        current: Tracklet | None = next((t for t in members if t.obs_id == current_key), None)
        active: list[Tracklet] = [t for t in members if t.is_active]
        chosen: Tracklet
        if not active:
            chosen = max(members, key=lambda t: t.last_active)
        elif current is None or not current.is_active:
            chosen = max(active, key=edge)
            if current is not None:
                self._handed_over_at[world_id] = now
        else:
            best: Tracklet = max(active, key=edge)
            held: bool = now - self._handed_over_at.get(world_id, -math.inf) < self._config.seam.hold
            beaten: bool = edge(best) >= edge(current) / self._config.seam.hysteresis
            chosen = best if best is not current and beaten and not held else current

        self._primary_for_world[world_id] = chosen.obs_id
        return chosen

    def prune(self, live_worlds: set[int]) -> None:
        """Forget the handover state of worlds that no longer exist."""
        for world_id in list(self._primary_for_world):
            if world_id not in live_worlds:
                del self._primary_for_world[world_id]
                self._handed_over_at.pop(world_id, None)

    # ── rules ──────────────────────────────────────────────────────────

    def _observations_match(self, a: Tracklet, b: Tracklet) -> bool:
        """True if two observations from DIFFERENT cameras are one person.

        **Azimuth** decides: within ``seam.link_angle`` degrees of world bearing. **Height** can veto,
        as a fraction of the larger measured height (``seam.link_height``), scale-free so two cameras
        at different distances agree — but only when both readings are measurements
        (`height_is_measured`), so a jump cannot refuse a link the azimuth supports.
        """
        if a.cam_id == b.cam_id:
            return False
        if not isinstance(a.annotation, Annotation) or not isinstance(b.annotation, Annotation):
            return False
        if self._rig.angle_diff(a.annotation.world_angle, b.annotation.world_angle) > self._config.seam.link_angle:
            return False
        return self._heights_match(a.annotation.height, b.annotation.height)

    def _heights_match(self, height_a: float, height_b: float) -> bool:
        """The scale-free height veto: `|a - b| / max(a, b)` against ``seam.link_height``, a
        fraction. Passes whenever either side has no usable reading — see `_observations_match`."""
        if not (height_is_measured(height_a) and height_is_measured(height_b)):
            return True
        largest: float = max(height_a, height_b)
        return abs(height_a - height_b) / largest <= self._config.seam.link_height

    def _collapse_pairs(self) -> list[tuple[int, int]]:
        """(keep_id, drop_id) pairs of worlds whose observations match across cameras; older world
        wins. Only mutual nearest matches, so an observation already explained by a partner in its
        own world cannot drag a neighbour in; each world in at most one pair."""
        observations: list[Tracklet] = [
            t for t in self._store.all_tracklets()
            if not t.is_removed
            and isinstance(t.annotation, Annotation)
            # Eligible where a second opinion exists at all: the picture's overlap, not a tuned
            # zone. Whether two eligible observations are one person is `_observations_match`.
            and self._rig.angle_in_overlap(t.annotation.local_angle)
        ]

        def nearest_match(t: Tracklet) -> Tracklet | None:
            best: Tracklet | None = None
            best_diff: float = float('inf')
            assert isinstance(t.annotation, Annotation)
            for o in observations:
                if o is t or not self._observations_match(t, o):
                    continue
                assert isinstance(o.annotation, Annotation)
                diff: float = self._rig.angle_diff(t.annotation.world_angle, o.annotation.world_angle)
                if diff < best_diff:
                    best_diff = diff
                    best = o
            return best

        nearest: list[Tracklet | None] = [nearest_match(t) for t in observations]

        used: set[int] = set()  # world ids already committed to a pair
        pairs: list[tuple[int, int]] = []
        for (i, a), (j, b) in combinations(enumerate(observations), 2):
            if a.id == b.id:
                continue
            if a.id in used or b.id in used:
                continue
            # A LOST observation may anchor a merge, but never merge two
            # worlds on lost data alone.
            if not (a.is_active or b.is_active):
                continue
            if nearest[i] is not b or nearest[j] is not a:
                continue
            if any(self._store.camera_sees_world(t.cam_id, b.id)
                   for t in self._store.get_tracklets(a.id) if t.is_active):
                continue

            # Older world wins
            members_a: list[Tracklet] = self._store.get_tracklets(a.id)
            members_b: list[Tracklet] = self._store.get_tracklets(b.id)
            oldest_a: float = min(t.created_at for t in members_a) if members_a else float('inf')
            oldest_b: float = min(t.created_at for t in members_b) if members_b else float('inf')
            keep_id, drop_id = (a.id, b.id) if oldest_a <= oldest_b else (b.id, a.id)
            used.add(a.id)
            used.add(b.id)
            pairs.append((keep_id, drop_id))
        return pairs
