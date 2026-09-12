"""One observation turned into one mark on the strip. No GL, no board, no settings — arithmetic.

Kept out of the renderers because two of them draw the same marks: `ObservationRenderer` draws the
lines, `LabelRenderer` the text beside them, and if each worked the geometry out for itself the two
could disagree about where a person is. The compositor builds the list once and hands it to both.

**A mark is the tracker's own belief, and all of it sits at one depth.** Its x is the fused
`world_angle` — the number the light, the sound and the hit detector all act on — and the tracker
derives that at a **fixed** depth, `rig`'s `parallax_diameter` (the harmonic mean of the tracked
zone, Ø 4.2 on this rig), never from a person's measured distance. See
`Geometry._update_parallax_depth` for why: the measured distance is biased by the device's box
bottom, and the two cameras at a seam err in opposite directions, so feeding it in cost about 10°
more than assuming a depth. The rows and the tolerance follow onto the same cylinder, so x and y
describe a person at one place rather than two.

Two things follow that a reader has to know:

- **A mark sits a small constant distance from its own pixels.** The image is stitched at
  `render.panorama.focus_diameter` (Ø 4.5) and the marks at Ø 4.2, so they are drawn on two nearby
  cylinders. That offset is a chosen consequence — the parallax depth is derived from the zone so it
  minimises the worst-case *seam* error, not so it matches a render slider — and it is not a fault.
- **Two marks of one colour at a seam no longer have to coincide exactly.** They disagree by the
  known geometric residual for that person's depth: zero at Ø 4.2, up to about 6.6° at the zone's
  edges. So the gap is now a **depth indicator**, not an error signal. It is a weaker diagnostic
  than the old one, and an honest one — the old one was reading the detector's bias.

`R` and `H` on the label are the only things here still derived from the measured distance, and they
are readouts, never placements. Both read low; `H` also *falls* as a person walks away, which is the
signature of a box bottom under the feet.

Each mark also carries the **joining tolerance that governs its place on the ring** as a width
(`_tolerance`), so the rule the tracker is about to apply to it is on screen next to it rather than
only in a settings panel.
"""

# Standard library imports
from dataclasses import dataclass

# Local application imports
from modules.tracker import PanoramicAnnotation, Tracklet, TrackingStatus, \
    camera_azimuth, camera_local_to_azimuth, centre_distance, centre_elevation, \
    elevation_from_row, focus_distance, strip_y, wrap180


@dataclass(frozen=True)
class StripGeometry:
    """The strip's geometry, in one value object, so a mark is one argument's worth of context.

    Owned by the compositor and rebuilt each tick from the tracker's published numbers; nothing
    here is a preference. `row_model` is (horizon_row, focal_rows) — the delivered frames' rows as
    the tracker published them — and `elevation_window` the strip's (top, bottom) at the rig centre.

    `parallax_diameter` is the one depth the tracker corrects the azimuth at, mirrored here so a
    mark's rows and its tolerance land on the same cylinder its x does.
    """
    cam_fov: float
    target_fov: float
    ring_radius: float
    parallax_diameter: float
    row_model: tuple[float, float]
    elevation_window: tuple[float, float]
    link_angle: float
    reacquire_angle: float

    @property
    def parallax_radius(self) -> float:
        return max(1e-6, self.parallax_diameter / 2.0)


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
    # The whole mark sits on the SAME cylinder the tracker corrects the azimuth at, so x, the rows
    # and the tolerance all describe a person at one depth. `phi` is the bearing at the rig centre
    # that this camera's column lands on there; `cylinder_distance` how far the camera is from the
    # cylinder along it — the stitch's own pair of numbers.
    phi: float = wrap180(annotation.world_angle - camera_azimuth(tracklet.cam_id, g.target_fov))
    cylinder_distance: float = focus_distance(phi, g.ring_radius, g.parallax_radius)

    top_y: float = _row_y(tracklet.roi.y, g, cylinder_distance)
    bottom_y: float = _row_y(tracklet.roi.y + tracklet.roi.height, g, cylinder_distance)

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
        # as the digits change. `R` and `H` are the ONLY things on the strip still derived from the
        # tracker's measured distance, and they are readouts, never placements — the mark's x, rows
        # and tolerance all come off the cylinder above. Both read **low**, because the device's box
        # bottom sits under the feet (a 1.8 m person prints H 1.1-1.4, and H falls as they walk
        # away, which is the signature of that cause). `R` is a radius from the rig centre where the
        # footer's Ø is a diameter — a factor of two apart, not the same kind of number.
        label=f'#{tracklet.id} c{tracklet.cam_id} '
              f'az{annotation.world_angle % 360.0:03.0f} '
              f'R{centre_distance(annotation.local_angle - g.cam_fov / 2.0, max(1e-6, annotation.distance), g.ring_radius):.1f}m '
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
    # azimuth goes through — `camera_local_to_azimuth` at `parallax_diameter`. That is what keeps
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
                                        g.target_fov, g.ring_radius, g.parallax_diameter)
    hi: float = camera_local_to_azimuth(min(g.cam_fov, local + half_local), cam_id, g.cam_fov,
                                        g.target_fov, g.ring_radius, g.parallax_diameter)
    return (lo, hi)


def _row_y(row: float, g: StripGeometry, cylinder_distance: float) -> float:
    """A normalised frame row to a normalised strip y.

    The delivered frame is cylindrical and levelled, so a row is the tangent of an elevation
    measured at the camera, below the horizon row (`elevation_from_row`). Converting that to the
    rig centre needs a distance, and it is the **cylinder's** at this bearing — the same depth the
    mark's x is corrected at — so the whole mark is one consistent statement at one depth. It used
    to be the person's own estimated distance, which put x and y on two different depths once the
    azimuth stopped using it.

    A row may legitimately fall outside [0, 1] — the device tracker extrapolates a partly visible
    person, and that is real information about how close they are — so nothing is clamped here; the
    renderer clips when it draws.
    """
    cam_elevation: float = elevation_from_row(row, *g.row_model)
    return strip_y(centre_elevation(cam_elevation, cylinder_distance, g.parallax_radius),
                   g.elevation_window)
