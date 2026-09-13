"""One observation turned into one mark on the strip. No GL, no board, no settings — arithmetic.

Kept out of the renderers because two of them draw the same marks: `ObservationRenderer` draws the
lines, `LabelRenderer` the text beside them, and if each worked the geometry out for itself the two
could disagree about where a person is. The compositor builds the list once and hands it to both.

**A mark is the tracker's own belief, and its two axes deliberately use two different depths.**
Both axes share the strip's space — x is azimuth at the rig centre, y is elevation there — and what
differs is the distance each is converted through, because each answers a different question.

**x: the fixed parallax depth.** The x is the fused `world_angle`, the number the light, the sound
and the hit detector all act on, and the tracker derives it at `rig.parallax_radius` (the tracked
zone's harmonic mean, R 2.1 here), never from a person's measured distance —
`Geometry._update_parallax_depth` has the measurements: the two cameras at a seam err in opposite
directions, so feeding a measured distance in cost about 10° more than assuming one. The tolerance
field follows x onto that same cylinder, because it is a statement about the same azimuth.

**y: the person's own distance.** The rows go through `annotation.distance` instead, and that is
what makes the foot row an *instrument* rather than a decoration. Converted through the person's own
distance the lens height cancels algebraically:

    atan(tan(-atan(h / d)) * d / R) = atan(-h / R)

which is *precisely* the formula the grid's zone field is drawn from. So the foot tick and the zone
lines are exactly comparable: **the tick sits on the R 3.5 line iff the tracker reports this person
at R 3.5** — tape the circle, stand on it, read it off, and the label's `R` prints that same 3.5.
Put the rows on the parallax cylinder for uniformity with x and that exactness is gone (20 px of
error at R 1.5), and the one calibration the strip can do precisely goes with it.

Note what this buys besides: the foot correction (`TrackerSettings.foot_offset`) reaches the mark
through `annotation.distance`, already computed, so **nothing here knows about it** and the
correction stays in one place. The distance is **unclamped**, so someone at R 5 reads `R5.0` with
the tick well above the zone field rather than piled up at its edge — which is what lets the strip
show why the tracker stopped counting them. A tick off the strip is not drawn
(`ObservationRenderer`).

Two consequences a reader has to know:

- **A mark sits a small constant distance from its own pixels.** The image is stitched at
  `render.panorama.focus_radius` (R 2.25) and a mark's x at R 2.1, so they are drawn on two nearby
  cylinders. That offset is a chosen consequence — the parallax depth is derived from the zone so it
  minimises the worst-case *seam* error, not so it matches a render slider — and it is not a fault.
- **Two marks of one colour at a seam no longer have to coincide exactly.** They disagree by the
  known geometric residual for that person's depth: zero at R 2.1, up to about 6.6° at the zone's
  edges. So the gap is a **depth indicator**, not an error signal. It is a weaker diagnostic than
  the old one, and an honest one — the old one was reading the detector's bias.

Each mark also carries the **joining tolerance that governs its place on the ring** as a width
(`_tolerance`), so the rule the tracker is about to apply to it is on screen next to it rather than
only in a settings panel.

**Detections a filter dropped are marks too**, so nobody leaves the strip without a reason. They
have no world, so they are grey, have no tolerance (no rule can join them), carry the detector's own
box — a box too small to count *looks* too small — and are labelled with the filter's name
(`Rejection`: `young`, `small`, `dead zone`, `past R3.5`). A tracked person the far edge stops
counting keeps their own mark, fading to grey over `lost_timeout` (`_color`); the moment it is fully
grey is the moment their grey box takes over.
"""

# Standard library imports
import math
from dataclasses import dataclass

# Local application imports
from modules.tracker import PanoramicAnnotation, Tracklet, TrackingStatus, camera_azimuth, \
    camera_local_to_azimuth, centre_distance, centre_elevation, elevation_from_row, focus_distance, \
    rejection_label, strip_y, wrap180


@dataclass(frozen=True)
class StripGeometry:
    """The strip's geometry, in one value object, so a mark is one argument's worth of context.

    Owned by the compositor and rebuilt each tick from the tracker's published numbers; nothing
    here is a preference. `row_model` is (horizon_row, focal_rows) — the delivered frames' rows,
    rebuilt from the tracker's published edge angles (`panorama_map.row_model`) — and
    `elevation_window` the strip's (top, bottom) at the rig centre.

    `parallax_radius` is the one depth the tracker corrects the azimuth at, mirrored here so a
    mark's x and its tolerance land on the same cylinder — the tracker's own number, unconverted.
    `camera_height` is the rig's measured lens height, and it is here for the *rows*: the foot tick
    is `atan(camera_height / R)` below the horizon, the grid's zone field's own formula, which is
    what makes the two comparable.

    `zone_max_radius` names the far edge in a `past R…` tag; `lost_timeout` and `now` are what a
    LOST mark's fade to grey is timed by — passed in rather than read, so this stays pure.
    """
    cam_fov: float
    target_fov: float
    ring_radius: float
    parallax_radius: float
    camera_height: float
    row_model: tuple[float, float]
    elevation_window: tuple[float, float]
    link_angle: float
    reacquire_angle: float
    zone_max_radius: float
    lost_timeout: float
    now: float


@dataclass(frozen=True)
class Mark:
    """Where one observation lands on the strip, and what to say about it."""
    world_id: int
    cam_id: int
    x: float                                        # normalised strip x, from the world azimuth
    top_y: float                                    # normalised strip y of the box's top row
    bottom_y: float                                 # ... and of the feet
    has_foot: bool                                  # whether `bottom_y` is a floor reading at all
    tolerance_x: float                              # left edge of the joining tolerance...
    tolerance_w: float                              # ... and its width; both normalised strip x
    box_x: float                                    # the detector's own box: left edge...
    box_w: float                                    # ... and width, normalised strip x...
    box_top_y: float                                # ... and its top and bottom rows
    box_bottom_y: float
    color: tuple[float, float, float, float]        # the world colour, alpha carrying confidence
    is_primary: bool
    rejected: bool                                  # a detection a filter dropped: no world, grey
    label: str


def build_marks(observations: list[Tracklet], primaries: set[int],
                colors: list[tuple[float, float, float, float]],
                rejected_color: tuple[float, float, float, float],
                geometry: StripGeometry) -> list[Mark]:
    """A mark per usable observation. Dropped detections first, under everyone; primaries last, so
    they are drawn over their candidates."""
    marks: list[Mark] = []
    for tracklet in observations:
        mark: Mark | None = _mark(tracklet, primaries, colors, rejected_color, geometry)
        if mark is not None:
            marks.append(mark)
    marks.sort(key=lambda m: (not m.rejected, m.is_primary))
    return marks


def _mark(tracklet: Tracklet, primaries: set[int],
          colors: list[tuple[float, float, float, float]],
          rejected_color: tuple[float, float, float, float],
          g: StripGeometry) -> Mark | None:
    if tracklet is None or tracklet.is_removed:
        return None
    if not isinstance(tracklet.annotation, PanoramicAnnotation):
        return None

    annotation: PanoramicAnnotation = tracklet.annotation
    roi_bottom: float = tracklet.roi.y + tracklet.roi.height
    # The ROWS go through the person's OWN distance, unlike the x above. `cam_distance` is how far
    # the camera reads them as being (through `foot_offset`, and unclamped); `centre_dist` the
    # radius from the rig centre the label prints as `R`, and the radius the foot tick is then exact
    # at against the zone field — so a tick above the field's top edge is a person past the far
    # edge, whom the tracker no longer sees.
    has_foot: bool = math.isfinite(annotation.distance)
    if has_foot:
        cam_distance: float = max(1e-6, annotation.distance)
        centre_dist: float = centre_distance(annotation.local_angle - g.cam_fov / 2.0,
                                             cam_distance, g.ring_radius)
        top_y: float = _row_y(tracklet.roi.y, g, cam_distance, centre_dist)
        box_bottom_y: float = _row_y(roi_bottom, g, cam_distance, centre_dist)
        bottom_y: float = _foot_y(g, centre_dist)
    else:
        # Feet at or above the horizon: not standing on this floor, so there is no distance to
        # convert the rows through. Place the box on the parallax cylinder instead — where its x
        # already is — and draw no foot tick, since there is no floor reading to mark.
        phi: float = wrap180(annotation.world_angle - camera_azimuth(tracklet.cam_id, g.target_fov))
        cylinder: float = focus_distance(phi, g.ring_radius, g.parallax_radius)
        centre_dist = math.inf
        top_y = _row_y(tracklet.roi.y, g, cylinder, g.parallax_radius)
        box_bottom_y = _row_y(roi_bottom, g, cylinder, g.parallax_radius)
        bottom_y = box_bottom_y

    # The detector's own box, across: its two edge columns through the same map the line takes.
    left: float = camera_local_to_azimuth(tracklet.roi.x * g.cam_fov, tracklet.cam_id, g.cam_fov,
                                          g.target_fov, g.ring_radius, g.parallax_radius)
    right: float = camera_local_to_azimuth((tracklet.roi.x + tracklet.roi.width) * g.cam_fov,
                                           tracklet.cam_id, g.cam_fov, g.target_fov,
                                           g.ring_radius, g.parallax_radius)

    # Dropped by a filter and never given a world: grey, no rule to join it by, tagged with why.
    rejected: bool = annotation.rejected is not None and tracklet.id < 0
    is_primary: bool = tracklet.obs_id in primaries
    if rejected:
        tolerance_lo = tolerance_hi = annotation.world_angle
        color: tuple[float, float, float, float] = rejected_color
    else:
        tolerance_lo, tolerance_hi = _tolerance(annotation, tracklet.cam_id, g)
        color = _color(tracklet, is_primary, colors, rejected_color, g)

    return Mark(
        world_id=tracklet.id,
        cam_id=tracklet.cam_id,
        x=(annotation.world_angle % 360.0) / 360.0,
        top_y=top_y,
        bottom_y=bottom_y,
        has_foot=has_foot,
        tolerance_x=(tolerance_lo % 360.0) / 360.0,
        tolerance_w=((tolerance_hi - tolerance_lo) % 360.0) / 360.0,
        box_x=(left % 360.0) / 360.0,
        box_w=((right - left) % 360.0) / 360.0,
        box_top_y=top_y,
        box_bottom_y=box_bottom_y,
        color=color,
        is_primary=is_primary,
        rejected=rejected,
        label=_label(tracklet, annotation, centre_dist, rejected, g),
    )


def _color(tracklet: Tracklet, is_primary: bool, colors: list[tuple[float, float, float, float]],
           grey: tuple[float, float, float, float], g: StripGeometry) -> tuple[float, float, float, float]:
    """The world colour, its alpha saying how much to trust it.

    A loser at a seam is dimmed so the primary reads as the one in charge. A **LOST** observation
    fades from the world colour to grey over `lost_timeout`, at the candidate's alpha: it is still
    an identity the tracker holds — anchoring a seam crossing, or a person walking past the far
    edge — and how grey it is says how close it is to being forgotten. Fully grey is exactly the
    moment a dropped detection's grey box takes over, so a person walking out never vanishes.
    """
    r, gr, b, a = colors[tracklet.id % len(colors)] if colors else (1.0, 1.0, 1.0, 1.0)
    if tracklet.status == TrackingStatus.LOST:
        age: float = max(0.0, g.now - tracklet.last_active)
        t: float = min(1.0, age / max(1e-6, g.lost_timeout))
        return (r + (grey[0] - r) * t, gr + (grey[1] - gr) * t, b + (grey[2] - b) * t, a * 0.5)
    return (r, gr, b, a if is_primary else a * 0.5)


def _label(tracklet: Tracklet, annotation: PanoramicAnnotation, centre_dist: float,
           rejected: bool, g: StripGeometry) -> str:
    """What to say beside a mark: the filter's name for a dropped detection, the readout otherwise.

    Fixed width, so the right-edge flip threshold is the same for everybody and cannot wobble as
    the digits change. `R` is the same radius the foot tick is drawn at — the tick is the picture of
    this number — so a tick on the R 3.5 zone line and an `R` of 3.5 say the same thing twice, and
    `rig.zone_max_radius` says 3.5 as well. Both `R` and `H` read low until
    `track.foot_offset` is measured; `H` falling as a person walks away is the signature
    that it has not been. A tracked person the far edge has stopped counting says so after it.
    """
    tag: str = '' if annotation.rejected is None else rejection_label(annotation.rejected, g.zone_max_radius)
    if rejected:
        return tag
    distance: str = f'R{centre_dist:.1f}m' if math.isfinite(centre_dist) else 'R-'
    label: str = (f'#{tracklet.id} c{tracklet.cam_id} '
                  f'az{annotation.world_angle % 360.0:03.0f} '
                  f'{distance} '
                  f'H{annotation.height:.1f}m')
    return f'{label} {tag}' if tag else label


def _tolerance(annotation: PanoramicAnnotation, cam_id: int,
               g: StripGeometry) -> tuple[float, float]:
    """(low, high) azimuth of the rule that decides what this observation may be joined to.

    **A pair test, so the field is the tolerance wide and not twice it.** Both gates have the form
    `|Δ| ≤ angle`, so two fields of `angle` touch exactly when the difference equals the gate: two
    fields of one colour that overlap are two observations the tracker will join. Fields of
    `±angle` would overlap out to twice the gate and claim links that never happen.

    **Which rule, and why the switch.** `overlap` means a second camera also sees this bearing, and
    there the question worth watching is the cross-camera one (`seam.link_angle`, in world azimuth).
    Outside it there is no second camera, so the only rule that can join anything is the same
    camera re-finding a person it dropped (`reacquire_angle`, in its own local angle). Both rules
    do in fact apply inside the overlap — a re-acquisition is tried first, everywhere — but the
    cross-camera one is the one being tuned, so it is the one drawn.
    """
    if annotation.overlap:
        # A world-azimuth gate: symmetric about the mark, and deliberately unclamped, since the
        # observation it might be joined to belongs to the camera whose field continues there.
        half: float = g.link_angle / 2.0
        centre: float = annotation.world_angle
        return (centre - half, centre + half)

    # A LOCAL-angle gate, carried into the strip's frame through the same map the tracker's own
    # azimuth goes through — `camera_local_to_azimuth` at `parallax_radius`. That is what keeps
    # the pair test valid once drawn: the two positions AND the two widths take the same map, so
    # their overlap still answers the gate, exactly rather than approximately (it used to depend on
    # each observation's own estimated distance, so two of them converted by slightly different
    # factors). It is also why the field measures visibly less than `reacquire_angle` against the
    # degree grid: the centre is further from the person than the camera is, so 5 deg of local angle
    # is about 4.5 deg of azimuth. Clamped to the camera's own field, because past its edge there
    # are no pixels for anyone to come back in through; the clamp cannot hide a real pair, since
    # both observations' centres lie inside the field and so does their overlap.
    half_local: float = g.reacquire_angle / 2.0
    local: float = annotation.local_angle
    lo: float = camera_local_to_azimuth(max(0.0, local - half_local), cam_id, g.cam_fov,
                                        g.target_fov, g.ring_radius, g.parallax_radius)
    hi: float = camera_local_to_azimuth(min(g.cam_fov, local + half_local), cam_id, g.cam_fov,
                                        g.target_fov, g.ring_radius, g.parallax_radius)
    return (lo, hi)


def _row_y(row: float, g: StripGeometry, cam_distance: float, centre_dist: float) -> float:
    """A normalised frame row to a normalised strip y, through THIS PERSON's own distance.

    The delivered frame is cylindrical and levelled, so a row is the tangent of an elevation
    measured at the camera, below the horizon row (`elevation_from_row`). Converting that to the rig
    centre needs a distance, and it is the person's own: only then is the mark's vertical extent
    their height on the same ruler the zone field is drawn on. Converting through the parallax
    cylinder instead — tidier, since x uses it — is what cost the foot tick its exactness.

    A row may legitimately fall outside [0, 1] — the device tracker extrapolates a partly visible
    person, and that is real information about how close they are — so nothing is clamped here; the
    renderer clips when it draws.
    """
    cam_elevation: float = elevation_from_row(row, *g.row_model)
    return strip_y(centre_elevation(cam_elevation, cam_distance, centre_dist), g.elevation_window)


def _foot_y(g: StripGeometry, centre_dist: float) -> float:
    """The strip y of the floor at radius `centre_dist` — the tick that reads against the zone.

    A closed form rather than `_row_y` of the box bottom, and it is the SAME number: running the
    foot row through the person's own distance gives

        atan(tan(-atan(h / d)) * d / R) = atan(-h / R)

    with the lens height `h` cancelling out of the conversion entirely. That right-hand side is
    exactly `GridRenderer._zone_field`'s formula, so the tick and the two zone edges are the same
    kind of number and may be compared by eye to the pixel. Written out rather than derived through
    the rows because it says what it means — *the floor, at the radius this person is reported at* —
    and because it cannot then drift from the zone line if the row model changes.

    The radius comes from the tracker's unclamped `distance`, so the tick keeps climbing toward the
    horizon as a person walks past the far edge. A box extrapolated far below the frame puts it
    below the strip, where `ObservationRenderer` does not draw it.
    """
    elevation: float = -math.degrees(math.atan(g.camera_height / max(1e-6, centre_dist)))
    return strip_y(elevation, g.elevation_window)
