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
correction stays in one place. The rows inherit the distance clamp for free, so a mangled box
cannot throw a tick off the strip.

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
"""

# Standard library imports
import math
from dataclasses import dataclass

# Local application imports
from modules.tracker import PanoramicAnnotation, Tracklet, TrackingStatus, \
    centre_distance, centre_elevation, elevation_from_row, strip_y, camera_local_to_azimuth


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


@dataclass(frozen=True)
class Mark:
    """Where one observation lands on the strip, and what to say about it."""
    world_id: int
    cam_id: int
    x: float                                        # normalised strip x, from the world azimuth
    top_y: float                                    # normalised strip y of the box's top row
    bottom_y: float                                 # ... and of its bottom row: the feet
    tolerance_x: float                              # left edge of the joining tolerance...
    tolerance_w: float                              # ... and its width; both normalised strip x
    color: tuple[float, float, float, float]        # the world colour, alpha carrying confidence
    is_primary: bool
    label: str


def build_marks(observations: list[Tracklet], primaries: set[int],
                colors: list[tuple[float, float, float, float]],
                geometry: StripGeometry) -> list[Mark]:
    """A mark per usable observation, primaries last so they are drawn over their candidates."""
    marks: list[Mark] = []
    for tracklet in observations:
        mark: Mark | None = _mark(tracklet, primaries, colors, geometry)
        if mark is not None:
            marks.append(mark)
    marks.sort(key=lambda m: m.is_primary)
    return marks


def _mark(tracklet: Tracklet, primaries: set[int],
          colors: list[tuple[float, float, float, float]],
          g: StripGeometry) -> Mark | None:
    if tracklet is None or tracklet.is_removed:
        return None
    if not isinstance(tracklet.annotation, PanoramicAnnotation):
        return None

    annotation: PanoramicAnnotation = tracklet.annotation
    # The ROWS go through the person's OWN distance, unlike the x above. `cam_distance` is how far
    # the camera reads them as being (already through `foot_offset` and the clamp); `centre_dist`
    # the radius from the rig centre the label prints as `R`, and the radius the foot tick is then
    # exact at against the zone field.
    cam_distance: float = max(1e-6, annotation.distance)
    centre_dist: float = centre_distance(annotation.local_angle - g.cam_fov / 2.0,
                                         cam_distance, g.ring_radius)

    top_y: float = _row_y(tracklet.roi.y, g, cam_distance, centre_dist)
    bottom_y: float = _foot_y(g, centre_dist)

    tolerance_lo, tolerance_hi = _tolerance(annotation, tracklet.cam_id, g)

    is_primary: bool = tracklet.obs_id in primaries
    r, gr, b, a = colors[tracklet.id % len(colors)] if colors else (1.0, 1.0, 1.0, 1.0)
    # The colour says who; the alpha says how much to trust it. A LOST observation is still
    # anchoring a seam crossing, so it is drawn, but faintly; a loser at a seam is dimmed so the
    # primary reads as the one in charge.
    if tracklet.status == TrackingStatus.LOST:
        a *= 0.25
    elif not is_primary:
        a *= 0.5

    return Mark(
        world_id=tracklet.id,
        cam_id=tracklet.cam_id,
        x=(annotation.world_angle % 360.0) / 360.0,
        top_y=top_y,
        bottom_y=bottom_y,
        tolerance_x=(tolerance_lo % 360.0) / 360.0,
        tolerance_w=((tolerance_hi - tolerance_lo) % 360.0) / 360.0,
        color=(r, gr, b, a),
        is_primary=is_primary,
        # Fixed width, so the right-edge flip threshold is the same for everybody and cannot wobble
        # as the digits change. `R` is the same radius the foot tick is drawn at — the tick is the
        # picture of this number — so a tick on the R 3.5 zone line and an `R` of 3.5 say the same
        # thing twice, and `rig.zone_max_radius` says 3.5 as well. Both `R` and `H` read low until
        # `camera.tracker.foot_offset` is measured; `H` falling as a person walks away is the
        # signature that it has not been.
        label=f'#{tracklet.id} c{tracklet.cam_id} '
              f'az{annotation.world_angle % 360.0:03.0f} '
              f'R{centre_dist:.1f}m '
              f'H{annotation.height:.1f}m',
    )


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

    The radius comes from the tracker's clamped `distance`, so an extrapolated or mangled box moves
    the tick to the edge of the tracked band rather than off the strip.
    """
    elevation: float = -math.degrees(math.atan(g.camera_height / max(1e-6, centre_dist)))
    return strip_y(elevation, g.elevation_window)
