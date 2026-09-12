"""The 360-degree panorama map: azimuth back to the column of one camera.

This is the **inverse** of the forward chain in `geometry.py`, and it lives here rather than in
the render so the two can never drift apart. The forward direction turns a camera column into a
world azimuth (`_calc_local_angle`, `_parallax_corrected_local`, `_calc_world_angle`); the stitch
needs the other direction — given an output column of a 360-degree strip, which column of which
camera does it show.

Every step inverts exactly, with no approximation:

`_calc_local_angle` is `x * cam_fov`, so `x = local / cam_fov`. That is only exact because the
delivered frame is **equirectangular** (`modules/oak/camera/definitions.py`,
`warp_mesh_points`): a column is one azimuth at every height. There is no distortion term
left to invert.

`_calc_world_angle` is an offset, so it subtracts.

`_parallax_corrected_local` is a triangle, and inverting it is where the depth comes in. The
camera sits `ring_radius` out from the rig centre, aimed radially outward. A point at camera
bearing ``theta`` and camera distance ``d`` is seen from the centre at bearing ``phi``, and the
law of sines on the centre/camera/person triangle gives

    sin(theta - phi) = ring_radius * sin(phi) / d      ->   theta = phi + asin(r * sin(phi) / d)

exact, not a small-angle expansion.

**But `d` is the one thing an image cannot know.** The tracker estimates it per person from where
their feet meet the floor; a stitch has no person, only pixels, so it assumes one depth for the
whole image: a **cylinder** of diameter ``focus_diameter`` around the rig, the middle of the play
zone. The distance from a camera to that cylinder varies with bearing, and that is `focus_distance`
below — the law of cosines with the ring radius. So the image is exactly aligned at the focus
diameter and drifts, by a bounded amount, nearer and farther. That is the design: see step 4.3 of
the plan. Nothing about a person, a box or a pose feeds it, so nothing can fool it.
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

    Falls out of `_calc_world_angle` at the frame centre: `target_fov * cam_id + cam_fov / 2 -
    fov_overlap` collapses to `target_fov * (cam_id + 0.5)`, independent of the lens. With four
    cameras the axes are at 45, 135, 225 and 315, and the sector boundaries they meet on — the
    seams — at 0, 90, 180 and 270.
    """
    return target_fov * (cam_id + 0.5)


def wrap180(angle: float) -> float:
    """An angle difference folded into [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def focus_distance(bearing: float, ring_radius: float, focus_radius: float) -> float:
    """Distance (m) from a camera to the focus cylinder, at `bearing` degrees off its axis.

    `bearing` is measured **at the rig centre**, which is what an output column of the panorama
    gives directly. The law of cosines on the centre/camera/cylinder triangle. Symmetric about the
    camera's axis, largest straight ahead, smallest to the sides.
    """
    b: float = math.radians(bearing)
    d2: float = ring_radius * ring_radius + focus_radius * focus_radius \
        - 2.0 * ring_radius * focus_radius * math.cos(b)
    return math.sqrt(max(0.0, d2))


def azimuth_to_camera_x(azimuth: float, cam_id: int, cam_fov: float, target_fov: float,
                        ring_radius: float, focus_diameter: float) -> float | None:
    """The normalized column of camera `cam_id` showing this world azimuth, or None.

    None means the azimuth falls outside that camera's field — the caller draws nothing for it.
    The test is on the **raw** camera bearing, which is the frame's real extent; the parallax
    re-projection moves the accepted band, so at a non-zero `ring_radius` a camera covers less of
    the cylinder than its bare field suggests.
    """
    phi: float = wrap180(azimuth - camera_azimuth(cam_id, target_fov))
    distance: float = focus_distance(phi, ring_radius, focus_diameter / 2.0)

    theta: float = phi
    if ring_radius > 0.0 and distance > 1e-9:
        ratio: float = ring_radius * math.sin(math.radians(phi)) / distance
        theta = phi + math.degrees(math.asin(max(-1.0, min(1.0, ratio))))

    local: float = theta + cam_fov / 2.0
    if local < -_EDGE_TOLERANCE or local > cam_fov + _EDGE_TOLERANCE:
        return None
    return max(0.0, min(1.0, local / cam_fov))


def camera_elevation(elevation: float, bearing: float, ring_radius: float,
                     focus_radius: float) -> float:
    """The elevation (degrees) a camera sees for a point the rig centre sees at `elevation`.

    The vertical half of the same triangle `azimuth_to_camera_x` solves horizontally, and it has
    to be solved too: a camera `ring_radius` out from the centre is *closer* to the near wall of
    the focus cylinder, so it sees the same standing person at a wider bearing AND a higher
    elevation. A point on the cylinder stands `h` above the lens plane at horizontal distance
    `focus_radius` from the centre and `d` from the camera, so

        tan(e_camera) = h / d  and  tan(e_centre) = h / focus_radius
        ->  tan(e_camera) = tan(e_centre) * focus_radius / d

    and the height cancels: only the ratio of the two horizontal distances survives. The factor
    is `R / (R - r)` straight ahead — 1.19 at Ø 4.5 — falling toward 1 at the sides, exactly
    matching the horizontal compression, which is why re-projecting one axis without the other
    leaves everything in the panorama too tall for its width.

    `bearing` is measured at the centre, as `focus_distance` takes it.
    """
    distance: float = focus_distance(bearing, ring_radius, focus_radius)
    if distance <= 1e-9:
        return elevation
    return math.degrees(math.atan(math.tan(math.radians(elevation)) * focus_radius / distance))


def centre_distance(bearing: float, cam_distance: float, ring_radius: float) -> float:
    """A person's horizontal distance from the RIG CENTRE (m).

    `bearing` is measured at the camera, off its own optical axis — what the tracker's
    `local_angle - cam_fov / 2` gives. The camera faces radially outward with the centre
    `ring_radius` behind it, so the person sits at `(d*cos(b) + r, d*sin(b))` from the centre. The
    same triangle `Geometry._parallax_corrected_local` solves for the bearing, solved here for the
    length instead.
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
    window (`ParallaxSettings`). Transcribed into `panoramicstitch.frag`; keep the two in step.
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
    """
    radius: float = max(1e-6, focus_radius)
    ratio: float = max(0.0, radius - ring_radius) / radius
    low, high = band
    return (math.degrees(math.atan(math.tan(math.radians(high)) * ratio)),
            math.degrees(math.atan(math.tan(math.radians(low)) * ratio)))


def panorama_coverage(azimuth: float, num_cameras: int, cam_fov: float, target_fov: float,
                      ring_radius: float, focus_diameter: float) -> int:
    """How many cameras see this azimuth — the divisor an averaging blend needs.

    At `ring_radius = 0` this is 2 within `fov_overlap` of every seam and 1 elsewhere. With the
    parallax correction on, the bands are narrower (a camera pushed outward covers less of the
    cylinder as measured from the centre) but the shape is the same, and it is never 0 as long as
    the cameras' fields sum past 360.
    """
    return sum(
        1 for cam_id in range(num_cameras)
        if azimuth_to_camera_x(azimuth, cam_id, cam_fov, target_fov,
                               ring_radius, focus_diameter) is not None
    )
