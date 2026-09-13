"""The 360-degree panorama map: azimuth to the column of one camera, and back.

**One pair of functions, used in both directions by both sides.** `camera_local_to_azimuth` is the
forward direction — a camera's own column to a world azimuth — and it is what `Geometry.calc_angle`
calls, so the *tracker's* azimuth and the *stitch's* placement are the same arithmetic rather than
two copies of it. `azimuth_to_camera_x` is its exact inverse, which is what the stitch needs: given
an output column of the strip, which column of which camera does it show. The round-trip tests
prove they invert, and that is the guard against the two drifting apart.

`local = x * cam_fov` exactly, because the delivered frame is **cylindrical**
(`modules/oak/camera/definitions.py`, `warp_mesh_points`): a column is one azimuth at every height.
There is no distortion term left to invert.

The rest is one triangle. The camera sits `ring_radius` out from the rig centre, aimed radially
outward. A point at camera bearing ``theta`` and camera distance ``d`` is seen from the centre at
bearing ``phi``, and the law of sines on the centre/camera/point triangle gives

    sin(theta - phi) = ring_radius * sin(phi) / d      ->   theta = phi + asin(r * sin(phi) / d)

exact, not a small-angle expansion.

**And `d` is assumed, never measured — by both sides, for the same reason.** A stitch has no person,
only pixels, so it must assume a depth: a **cylinder** of radius ``focus_radius`` around the rig.
The tracker *could* measure a person's distance, and deliberately does not: the device's box bottom
sits below the feet, and the two cameras at a seam err in opposite directions, so feeding that in
cost about 10° more seam disagreement than assuming a depth (`Geometry._update_parallax_depth` has
the table). So the tracker assumes one too — its own `parallax_radius`, the tracked zone's
harmonic mean. Both are exactly aligned at their assumed radius and drift by a bounded amount
nearer and farther. Nothing about a person, a box or a pose feeds either, so nothing can fool them.

**Everything here is a RADIUS from the fixture axis, and nothing is ever halved.** The installation
is built and taped from the light fixture at the centre, so that is the origin; the settings, the
shader's ``focusRadius``/``ringRadius``, the panorama's ``R`` label and a tape on the floor all
carry the same number into these triangles, with no conversion anywhere to get wrong.

**The depth parameters are named `depth_radius` and not after any one caller.** These functions
serve every caller's depth — the stitch's `focus_radius`, the tracker's `parallax_radius`, the
zone's far edge, a person's own measured distance — and naming the parameter after one of them is
how a call ends up drawn at a depth its author did not mean. The rule the whole strip rests on:
**two things on the strip are comparable only if they share an axis *and* a depth.** Sharing the
axis is automatic (x is centre azimuth, y is centre elevation, everywhere); sharing the depth is
the caller's job. Every one of the panorama's three alignment bugs was two things at two depths,
never two things on two axes.

`elevation_window` is the exception and keeps `focus_radius`: it sets the strip's *single* y scale,
at the picture's depth, for everything drawn on it.
"""

import math

from modules.oak import FrameWindow


# The field test is a closed interval, and a column that lands exactly on the frame edge — the
# seam-most pixel of a camera, which is precisely where the stitch is read — comes out of the
# `asin` a float wobble past it. Admit that wobble and clamp, rather than dropping the one column
# the whole display exists to compare. A degree here is a thousandth of a pixel.
_EDGE_TOLERANCE: float = 1e-6


def fov_overlap(cam_fov: float, target_fov: float) -> float:
    """Half the field each camera shares with a neighbour (degrees) — `Geometry.fov_overlap`."""
    return (cam_fov - target_fov) / 2.0


def camera_azimuth(cam_id: int, target_fov: float) -> float:
    """World azimuth (degrees) a camera's optical axis points at.

    Falls out of the azimuth frame's own definition at the frame centre:
    `target_fov * cam_id + cam_fov / 2 - fov_overlap` collapses to `target_fov * (cam_id + 0.5)`,
    independent of the lens. With four
    cameras the axes are at 45, 135, 225 and 315, and the sector boundaries they meet on — the
    seams — at 0, 90, 180 and 270.
    """
    return target_fov * (cam_id + 0.5)


def wrap180(angle: float) -> float:
    """An angle difference folded into [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def focus_distance(bearing: float, ring_radius: float, depth_radius: float) -> float:
    """Distance (m) from a camera to a cylinder of radius `depth_radius`, `bearing` off its axis.

    The depth is the **caller's** choice — the picture's, the tracker's, the zone's — and this
    answers for whichever one it is handed.

    `bearing` is measured **at the rig centre**, which is what an output column of the panorama
    gives directly. The law of cosines on the centre/camera/cylinder triangle. Symmetric about the
    camera's axis, and **smallest straight ahead** — the camera is pushed toward the wall it faces,
    so its own axis is the short ray: `depth_radius - ring_radius` dead ahead against
    `+ ring_radius` behind (`test_closest_straight_ahead_farthest_behind`).
    """
    b: float = math.radians(bearing)
    d2: float = ring_radius * ring_radius + depth_radius * depth_radius \
        - 2.0 * ring_radius * depth_radius * math.cos(b)
    return math.sqrt(max(0.0, d2))


def azimuth_to_camera_x(azimuth: float, cam_id: int, cam_fov: float, target_fov: float,
                        ring_radius: float, depth_radius: float) -> float | None:
    """The normalized column of camera `cam_id` showing this world azimuth, or None.

    None means the azimuth falls outside that camera's field — the caller draws nothing for it.
    The test is on the **raw** camera bearing, which is the frame's real extent; the parallax
    re-projection moves the accepted band, so at a non-zero `ring_radius` a camera covers less of
    the cylinder than its bare field suggests.

    `depth_radius` is the caller's assumed depth, and the answer is only comparable with another
    call's at the same one.
    """
    phi: float = wrap180(azimuth - camera_azimuth(cam_id, target_fov))
    distance: float = focus_distance(phi, ring_radius, depth_radius)

    theta: float = phi
    if ring_radius > 0.0 and distance > 1e-9:
        ratio: float = ring_radius * math.sin(math.radians(phi)) / distance
        theta = phi + math.degrees(math.asin(max(-1.0, min(1.0, ratio))))

    local: float = theta + cam_fov / 2.0
    if local < -_EDGE_TOLERANCE or local > cam_fov + _EDGE_TOLERANCE:
        return None
    return max(0.0, min(1.0, local / cam_fov))


def camera_local_to_azimuth(local: float, cam_id: int, cam_fov: float, target_fov: float,
                            ring_radius: float, depth_radius: float) -> float:
    """The world azimuth a camera's own column points at, **at the depth the caller names**.

    The inverse of `azimuth_to_camera_x` (times `cam_fov`), and the one thing that lets the
    display draw a camera-frame quantity — a field edge, a dead zone, a lost person's last local
    angle — **where its pixels actually are**. Drawn at `camera_azimuth + local - cam_fov/2`
    instead, a band would sit up to several degrees away from the picture it describes, which is
    exactly the mistake the parallax correction exists to remove.

    The camera sits `ring_radius` out along its own axis, so a point at camera bearing
    `θ = local - cam_fov/2` on the focus cylinder of radius `R` is `d` away, and

        d = -r·cos θ + √(R² - r²·sin² θ)        (the cylinder, by the law of cosines)
        φ = atan2(d·sin θ, d·cos θ + r)          (that point's bearing from the centre)

    At `ring_radius = 0` this collapses to `φ = θ` and the whole thing to the plain offset
    `target_fov * cam_id + local - fov_overlap` that defines the azimuth frame. **This is the
    tracker's forward chain as well as the stitch's**, and the three live callers name three
    different depths: `Geometry.calc_angle` and the marks' tolerance fields at `parallax_radius`,
    `Geometry._update_overlap_band` at the zone's far edge, `SeamRenderer` at `focus_radius`.
    Two of those answers may not be compared with each other — the same local angle lands 2.3°
    apart at the studio's two depths — which is why the parameter is `depth_radius` and not any
    one of their names. No caller passes a person's measured distance, deliberately.
    """
    theta: float = math.radians(local - cam_fov / 2.0)
    phi: float = local - cam_fov / 2.0
    radius: float = max(0.0, depth_radius)
    if ring_radius > 0.0 and radius > 0.0:
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        root: float = radius * radius - ring_radius * ring_radius * sin_t * sin_t
        distance: float = -ring_radius * cos_t + math.sqrt(max(0.0, root))
        if distance > 1e-9:
            phi = math.degrees(math.atan2(distance * sin_t, distance * cos_t + ring_radius))
    return (camera_azimuth(cam_id, target_fov) + phi) % 360.0


def strip_spans(x: float, width: float) -> list[tuple[float, float]]:
    """Normalised x spans of a band `width` wide with its LEFT edge at `x`, wrapped at the join.

    The strip's left and right edges are the same azimuth, so a band that runs off one comes back
    on the other and has to be drawn as two quads. One function because four things need it — the
    seam bands, the two tolerance bars and the foot tick — and a band silently clipped at azimuth
    0 is invisible precisely where two cameras meet, which is where a reader is looking hardest.

    `x` may be any real number (a band centred just past 0 starts negative); a `width` at or past
    the whole strip is one full span, and a non-positive one draws nothing.
    """
    if width <= 0.0:
        return []
    if width >= 1.0:
        return [(0.0, 1.0)]
    left: float = x % 1.0
    if left + width <= 1.0:
        return [(left, width)]
    return [(left, 1.0 - left), (0.0, left + width - 1.0)]


def camera_elevation(elevation: float, bearing: float, ring_radius: float,
                     depth_radius: float) -> float:
    """The elevation (degrees) a camera sees for a point the rig centre sees at `elevation`.

    The vertical half of the same triangle `azimuth_to_camera_x` solves horizontally, and it has
    to be solved too: a camera `ring_radius` out from the centre is *closer* to the near wall of
    the assumed cylinder, so it sees the same standing person at a wider bearing AND a higher
    elevation. A point on that cylinder stands `h` above the lens plane at horizontal distance
    `depth_radius` from the centre and `d` from the camera, so

        tan(e_camera) = h / d  and  tan(e_centre) = h / depth_radius
        ->  tan(e_camera) = tan(e_centre) * depth_radius / d

    and the height cancels: only the ratio of the two horizontal distances survives. The factor
    is `R / (R - r)` straight ahead — 1.19 at R 2.25 — falling toward 1 at the sides, exactly
    matching the horizontal compression, which is why re-projecting one axis without the other
    leaves everything in the panorama too tall for its width.

    `depth_radius` is the caller's depth, as everywhere here; `bearing` is measured at the centre,
    as `focus_distance` takes it.
    """
    distance: float = focus_distance(bearing, ring_radius, depth_radius)
    if distance <= 1e-9:
        return elevation
    return math.degrees(math.atan(math.tan(math.radians(elevation)) * depth_radius / distance))


def centre_distance(bearing: float, cam_distance: float, ring_radius: float) -> float:
    """A person's horizontal distance from the RIG CENTRE (m).

    `bearing` is measured at the camera, off its own optical axis — what the tracker's
    `local_angle - cam_fov / 2` gives. The camera faces radially outward with the centre
    `ring_radius` behind it, so the person sits at `(d*cos(b) + r, d*sin(b))` from the centre. The
    same triangle `camera_local_to_azimuth` solves for the bearing, solved here for the length
    instead. Used only for the panorama label's `R` — a readout, never a placement.
    """
    theta: float = math.radians(bearing)
    x: float = cam_distance * math.cos(theta) + ring_radius
    y: float = cam_distance * math.sin(theta)
    return math.hypot(x, y)


def centre_elevation(cam_elevation: float, cam_distance: float, centre_dist: float) -> float:
    """The elevation (degrees) the rig centre sees for a point a camera sees at `cam_elevation`.

    The inverse of `camera_elevation`, for a point whose distance is actually known — a *person*,
    measured by the tracker, rather than the focus cylinder an image has to assume. One height, two
    horizontal distances, so the height cancels exactly as it does there:

        tan(e_centre) = tan(e_camera) * cam_distance / centre_dist

    Drawing a person through this and the image through `camera_elevation` is deliberate: the data
    lands where the tracker believes the person is, the pixels where the cylinder says, and a
    vertical gap between the two is the distance model being wrong.
    """
    if centre_dist <= 1e-9:
        return cam_elevation
    return math.degrees(math.atan(
        math.tan(math.radians(cam_elevation)) * cam_distance / centre_dist))


def populated_band(window: FrameWindow) -> tuple[float, float]:
    """(low, high) elevations the delivered frames carry, measured AT THE CAMERA.

    The frame's window: its bottom row is the sensor's lowest reach on the centre column and its
    top row whatever the rows reach. Off the centre column the sensor reaches less at the top
    (the black arch), and that is in the pixels as black rather than in this band; a stitch that
    counts coverage exactly per column would need `frame_coverage` (see CALIBRATION.md, *Open*).
    """
    return (window.elevation_bottom, window.elevation_top)


def row_from_elevation(elevation: float, horizon_row: float, focal_rows: float) -> float:
    """Normalised frame row (0 = top) of an elevation (degrees) at the camera.

    The delivered frame's rows are the tangent of elevation: `row = horizon_row - focal_rows *
    tan(e)`, with `horizon_row` the normalised row of elevation 0 (which may fall outside 0..1)
    and `focal_rows` the focal length in frame heights. Both come off the tracker's published
    window (`RigSettings`). Transcribed into `panoramicstitch.frag`; keep the two in step.
    """
    return horizon_row - focal_rows * math.tan(math.radians(elevation))


def elevation_from_row(row: float, horizon_row: float, focal_rows: float) -> float:
    """The inverse of `row_from_elevation`: a normalised frame row to an elevation (degrees)."""
    if focal_rows <= 0.0:
        return 0.0
    return math.degrees(math.atan((horizon_row - row) / focal_rows))


def strip_y(elevation: float, elevation_window: tuple[float, float]) -> float:
    """Normalised, top-down y of a centre elevation (degrees) in the 360-degree STRIP.

    The strip's rows are tangents of elevation, like the camera frames' — a photograph's vertical,
    not a degree scale — so a person or a ceiling edge has the same shape in the strip as in the
    frames, and the vertical parallax term is a plain scale per column. `elevation_window` is the
    strip's (top, bottom) at the rig centre (`elevation_window`). Transcribed into
    `panoramicstitch.frag`; keep the two in step.
    """
    top, bottom = elevation_window
    tan_top, tan_bottom = math.tan(math.radians(top)), math.tan(math.radians(bottom))
    span: float = tan_top - tan_bottom
    if abs(span) < 1e-9:
        return 0.5
    return (tan_top - math.tan(math.radians(elevation))) / span


def strip_elevation(y: float, elevation_window: tuple[float, float]) -> float:
    """The inverse of `strip_y`: a normalised strip y to a centre elevation (degrees)."""
    top, bottom = elevation_window
    tan_top, tan_bottom = math.tan(math.radians(top)), math.tan(math.radians(bottom))
    return math.degrees(math.atan(tan_top - y * (tan_top - tan_bottom)))


def strip_aspect_ratio(elevation_window: tuple[float, float]) -> float:
    """Width over height of the strip: 360 degrees of azimuth at the same focal as the tangent
    rows, so a degree at the horizon is the same size either way and the rows above spend more,
    exactly as the camera frames do. `2π / (tan(top) − tan(bottom))`."""
    top, bottom = elevation_window
    span: float = math.tan(math.radians(top)) - math.tan(math.radians(bottom))
    return 2.0 * math.pi / max(1e-6, span)


def elevation_window(band: tuple[float, float], ring_radius: float,
                     focus_radius: float) -> tuple[float, float]:
    """(top, bottom) elevation of the 360-degree strip, measured AT THE RIG CENTRE.

    The populated band converted to the centre's point of view, at the bearing where the conversion
    is tightest. `tan(e_centre) = tan(e_cam) * d / focus_radius`, and `d` is smallest straight ahead
    (`focus_radius - ring_radius`), so taking the window there guarantees every column of the strip
    is filled rather than fading to black near the camera axes.

    **`focus_radius`, not `depth_radius`** — the one function here that names its depth, because it
    has one caller and one meaning: it fixes the strip's **single, shared** y scale at the
    picture's depth, and everything drawn on the strip is then placed on that same ruler. So it
    moves the whole strip together and can never break a comparison between two things on it: drag
    `focus_radius` and the horizon stays at y 0.7791, because the scale and the thing measured
    move as one. That is the opposite of the per-call depths above, which do break comparisons.
    """
    radius: float = max(1e-6, focus_radius)
    ratio: float = max(0.0, radius - ring_radius) / radius
    low, high = band
    return (math.degrees(math.atan(math.tan(math.radians(high)) * ratio)),
            math.degrees(math.atan(math.tan(math.radians(low)) * ratio)))


def panorama_coverage(azimuth: float, num_cameras: int, cam_fov: float, target_fov: float,
                      ring_radius: float, depth_radius: float) -> int:
    """How many cameras see this azimuth — the divisor an averaging blend needs.

    At `ring_radius = 0` this is 2 within `fov_overlap` of every seam and 1 elsewhere. With the
    parallax correction on, the bands are narrower (a camera pushed outward covers less of the
    cylinder as measured from the centre) but the shape is the same, and it is never 0 as long as
    the cameras' fields sum past 360.
    """
    return sum(
        1 for cam_id in range(num_cameras)
        if azimuth_to_camera_x(azimuth, cam_id, cam_fov, target_fov,
                               ring_radius, depth_radius) is not None
    )
