"""One observation turned into one mark on the strip. No GL, no board, no settings — arithmetic.

Kept out of the renderers because two of them draw the same marks: `ObservationRenderer` draws the
lines, `LabelRenderer` the text beside them, and if each worked the geometry out for itself the two
could disagree about where a person is. The compositor builds the list once and hands it to both.

**A mark is the tracker's own belief, and nothing else.** Its x is the fused `world_angle` — the
number the light, the sound and the hit detector all act on — and its rows are re-projected to the
rig centre through that person's own estimated distance. The image underneath is placed where the
*focus cylinder* says (`camera_elevation`), so a mark generally does **not** sit on its own pixels,
and that displacement is not a measurement of anything: it is the difference between two depth
assumptions, zero only for a person standing at `focus_diameter` and zero on a camera axis at any
depth. The person's distance is already printed on the label as `R`, which is the honest way to
read it.

What the marks *are* for is the one check nothing else in the app can show: **two marks of one
colour, at a seam, must coincide.** Each camera corrects through its own distance estimate, so that
holds at any depth, and the gap between them is the fused azimuth being wrong — `fov`, `tilt`,
`ring_radius`, or the distance that feeds the correction.

Each mark also carries the **joining tolerance that governs its place on the ring** as a width
(`_tolerance`), so the rule the tracker is about to apply to it is on screen next to it rather than
only in a settings panel.
"""

# Standard library imports
from dataclasses import dataclass

# Local application imports
from modules.tracker import PanoramicAnnotation, Tracklet, TrackingStatus, \
    camera_azimuth, centre_bearing, centre_distance, centre_elevation, elevation_from_row, strip_y


@dataclass(frozen=True)
class StripGeometry:
    """The strip's geometry, in one value object, so a mark is one argument's worth of context.

    Owned by the compositor and rebuilt each tick from the tracker's published numbers; nothing
    here is a preference. `row_model` is (horizon_row, focal_rows) — the delivered frames' rows as
    the tracker published them — and `elevation_window` the strip's (top, bottom) at the rig centre.
    """
    cam_fov: float
    target_fov: float
    ring_radius: float
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
    # The bearing off this camera's own axis, which is what the parallax triangle takes.
    bearing: float = annotation.local_angle - g.cam_fov / 2.0
    cam_distance: float = max(1e-6, annotation.distance)
    centre_dist: float = centre_distance(bearing, cam_distance, g.ring_radius)

    top_y: float = _row_y(tracklet.roi.y, g, cam_distance, centre_dist)
    bottom_y: float = _row_y(tracklet.roi.y + tracklet.roi.height, g, cam_distance, centre_dist)

    tolerance_lo, tolerance_hi = _tolerance(annotation, tracklet.cam_id, bearing, cam_distance, g)

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
        # as the digits change. `R` is the drafting radius: this distance is from the rig centre,
        # where the footer's `Ø` is a diameter — the two differ by a factor of two and must not be
        # read as the same kind of number. `H` is the person's own height, measured at the camera
        # and so already absolute: a seam's two observations must print the same `H` while their
        # box heights in pixels do not, which is the one number on the strip that reads as a check
        # on itself, and the quantity `seam.link_height` gates on.
        label=f'#{tracklet.id} c{tracklet.cam_id} '
              f'az{annotation.world_angle % 360.0:03.0f} R{centre_dist:.1f}m '
              f'H{annotation.height:.1f}m',
    )


def _tolerance(annotation: PanoramicAnnotation, cam_id: int, bearing: float, cam_distance: float,
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

    # A LOCAL-angle gate, so it is carried into the strip's frame through the same triangle the
    # tracker corrected the person with — `centre_bearing` at their own distance, not the focus
    # cylinder's. That conversion is what keeps the pair test valid once drawn: the two positions
    # AND the two widths all go through the same map, so their overlap still answers the gate. It
    # is why the field measures visibly less than `reacquire_angle` against the degree grid — the
    # centre is further from the person than the camera is, by `d / (d + r)` on axis, so 5 deg of
    # local angle is 4.5 deg of azimuth at 3 m and 4.7 at 6. (Two observations at different
    # estimated distances convert by slightly different factors, which makes the test approximate
    # by a fraction of a degree.) Clamped to the camera's own field, because past its edge there
    # are no pixels for anyone to come back in through; the clamp cannot hide a real pair, since
    # both observations' centres lie inside the field and so does their overlap.
    axis: float = camera_azimuth(cam_id, g.target_fov)
    half_local: float = g.reacquire_angle / 2.0
    lo: float = centre_bearing(max(-g.cam_fov / 2.0, bearing - half_local), cam_distance,
                               g.ring_radius)
    hi: float = centre_bearing(min(g.cam_fov / 2.0, bearing + half_local), cam_distance,
                               g.ring_radius)
    return (axis + lo, axis + hi)


def _row_y(row: float, g: StripGeometry, cam_distance: float, centre_dist: float) -> float:
    """A normalised frame row to a normalised strip y.

    The delivered frame is cylindrical and levelled, so a row is the tangent of an elevation
    measured at the camera, below the horizon row (`elevation_from_row`). Converting that to the
    rig centre uses the **person's own** distance, as the x does, so the whole mark is one
    consistent statement of where the tracker believes they are.

    A row may legitimately fall outside [0, 1] — the device tracker extrapolates a partly visible
    person, and that is real information about how close they are — so nothing is clamped here; the
    renderer clips when it draws.
    """
    cam_elevation: float = elevation_from_row(row, *g.row_model)
    return strip_y(centre_elevation(cam_elevation, cam_distance, centre_dist), g.elevation_window)
