"""Projection between a camera's own frame and the world seen from the fixture axis.

A camera's column (local angle) to world azimuth and back, camera distance to centre distance, and
the frame's rows to elevation and back. `camera_local_to_azimuth` (forward, what `Rig.calc_angle`
uses) and `azimuth_to_camera_x` (inverse, what the stitch uses) are one triangle, so the tracker's
azimuths and the stitch's placement are the same arithmetic; the round-trip tests guard that.
`local = x * cam_fov` exactly, because the delivered frame is cylindrical.

The camera sits `camera_radius` out from the fixture axis, facing outward. A point at camera bearing
θ and camera distance d is seen from the centre at φ, by the law of sines:

    sin(θ - φ) = r · sin(φ) / d

**The depth d is always assumed, never a person's measurement.** The stitch assumes a cylinder at
`focus_radius`; the tracker assumes its `parallax_radius` (why: `Rig._update_parallax_depth`). So
the depth parameter is `depth_radius`, named after no caller: two results on the strip are only
comparable if they were computed at the same depth, and that is the caller's job.

**Everything is a radius from the fixture axis**, the same number the settings, the shader, the
panorama's `R` label and a tape on the floor carry. The strip's own rows and extent live with the
display (`modules/render/layers/panorama/strip.py`).
"""

import math


# The field test is a closed interval, and a column that lands exactly on the frame edge — the
# seam-most pixel of a camera, which is precisely where the stitch is read — comes out of the
# `asin` a float wobble past it. Admit that wobble and clamp, rather than dropping the one column
# the whole display exists to compare. A degree here is a thousandth of a pixel.
_EDGE_TOLERANCE: float = 1e-6


def camera_azimuth(cam_id: int, target_fov: float) -> float:
    """World azimuth (degrees) a camera's optical axis points at.

    Falls out of the azimuth frame's own definition at the frame centre:
    `target_fov * cam_id + cam_fov / 2 - (cam_fov - target_fov) / 2` collapses to
    `target_fov * (cam_id + 0.5)`, independent of the lens. With four
    cameras the axes are at 45, 135, 225 and 315, and the sector boundaries they meet on — the
    seams — at 0, 90, 180 and 270.
    """
    return target_fov * (cam_id + 0.5)


def wrap180(angle: float) -> float:
    """An angle difference folded into [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def focus_distance(bearing: float, camera_radius: float, depth_radius: float) -> float:
    """Distance (m) from a camera to a cylinder of radius `depth_radius`, `bearing` off its axis.

    The depth is the **caller's** choice — the picture's, the tracker's, the zone's — and this
    answers for whichever one it is handed.

    `bearing` is measured **at the rig centre**, which is what an output column of the panorama
    gives directly. The law of cosines on the centre/camera/cylinder triangle. Symmetric about the
    camera's axis, and **smallest straight ahead** — the camera is pushed toward the wall it faces,
    so its own axis is the short ray: `depth_radius - camera_radius` dead ahead against
    `+ camera_radius` behind (`test_closest_straight_ahead_farthest_behind`).
    """
    b: float = math.radians(bearing)
    d2: float = camera_radius * camera_radius + depth_radius * depth_radius \
        - 2.0 * camera_radius * depth_radius * math.cos(b)
    return math.sqrt(max(0.0, d2))


def camera_bearing(centre_bearing: float, camera_radius: float, distance: float) -> float:
    """The bearing (degrees) off a camera's own axis of a point the centre sees `centre_bearing`
    off that axis, `distance` m from the camera.

    Wider than the centre's, because the camera sits `camera_radius` out toward the point:
    `theta = phi + asin(r · sin(phi) / d)`, the law of sines on the centre/camera/point triangle.
    """
    if camera_radius <= 0.0 or distance <= 1e-9:
        return centre_bearing
    ratio: float = camera_radius * math.sin(math.radians(centre_bearing)) / distance
    return centre_bearing + math.degrees(math.asin(max(-1.0, min(1.0, ratio))))


def azimuth_to_camera_x(azimuth: float, cam_id: int, cam_fov: float, target_fov: float,
                        camera_radius: float, depth_radius: float) -> float | None:
    """The normalized column of camera `cam_id` showing this world azimuth, or None.

    None means the azimuth falls outside that camera's field — the caller draws nothing for it.
    The test is on the **raw** camera bearing, which is the frame's real extent; the parallax
    re-projection moves the accepted band, so at a non-zero `camera_radius` a camera covers less of
    the cylinder than its bare field suggests.

    `depth_radius` is the caller's assumed depth, and the answer is only comparable with another
    call's at the same one.
    """
    phi: float = wrap180(azimuth - camera_azimuth(cam_id, target_fov))
    distance: float = focus_distance(phi, camera_radius, depth_radius)
    theta: float = camera_bearing(phi, camera_radius, distance)

    local: float = theta + cam_fov / 2.0
    if local < -_EDGE_TOLERANCE or local > cam_fov + _EDGE_TOLERANCE:
        return None
    return max(0.0, min(1.0, local / cam_fov))


def camera_local_to_azimuth(local: float, cam_id: int, cam_fov: float, target_fov: float,
                            camera_radius: float, depth_radius: float) -> float:
    """The world azimuth a camera's own column (`local`, degrees) points at, at `depth_radius`.

    The inverse of `azimuth_to_camera_x` (times `cam_fov`): the tracker's forward chain, and how
    the display draws a camera-frame quantity (a field edge, a dead zone) where its pixels are.
    A point at camera bearing `θ = local - cam_fov/2` on the cylinder of radius `R` is `d` away:

        d = -r·cos θ + √(R² - r²·sin² θ)        (law of cosines)
        φ = atan2(d·sin θ, d·cos θ + r)          (its bearing from the centre)

    At `camera_radius = 0` this is the plain offset `target_fov * cam_id + local -
    (cam_fov - target_fov) / 2` that defines the azimuth frame. Callers use different depths —
    `parallax_radius` (tracker, marks), the zone's far edge (overlap band), `focus_radius` (seam
    bands) — and results at different depths are not comparable.
    """
    theta: float = math.radians(local - cam_fov / 2.0)
    phi: float = local - cam_fov / 2.0
    radius: float = max(0.0, depth_radius)
    if camera_radius > 0.0 and radius > 0.0:
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        root: float = radius * radius - camera_radius * camera_radius * sin_t * sin_t
        distance: float = -camera_radius * cos_t + math.sqrt(max(0.0, root))
        if distance > 1e-9:
            phi = math.degrees(math.atan2(distance * sin_t, distance * cos_t + camera_radius))
    return (camera_azimuth(cam_id, target_fov) + phi) % 360.0


def centre_distance(bearing: float, cam_distance: float, camera_radius: float) -> float:
    """A person's horizontal distance from the RIG CENTRE (m).

    `bearing` is measured at the camera, off its own optical axis — what the tracker's
    `local_angle - cam_fov / 2` gives. The camera faces radially outward with the centre
    `camera_radius` behind it, so the person sits at `(d*cos(b) + r, d*sin(b))` from the centre. The
    same triangle `camera_local_to_azimuth` solves for the bearing, solved here for the length
    instead. Used for the far-edge test, the panorama label's `R` and the pose's `Distance` —
    never a placement.
    """
    theta: float = math.radians(bearing)
    x: float = cam_distance * math.cos(theta) + camera_radius
    y: float = cam_distance * math.sin(theta)
    return math.hypot(x, y)


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


def row_model(angle_bottom: float, angle_top: float) -> tuple[float, float]:
    """(horizon_row, focal_rows) of a frame whose bottom row is at `angle_bottom` and top row at
    `angle_top` (degrees from eye level) — the row model, rebuilt from the frame's two edges.

    Exact, not an approximation: the two numbers carry no information the angles don't. At row 0
    `tan(top) = horizon_row / focal_rows`, at row 1 `tan(bottom) = (horizon_row - 1) / focal_rows`,
    so `focal_rows = 1 / (tan(top) - tan(bottom))` and `horizon_row = focal_rows · tan(top)`. That
    is why the tracker publishes only the angles — readable, and in the panel's own units — and the
    stitch and the marks derive the row form here where they need it.
    """
    tan_top: float = math.tan(math.radians(angle_top))
    span: float = tan_top - math.tan(math.radians(angle_bottom))
    if span < 1e-9:
        return 0.5, 1e-6
    focal_rows: float = 1.0 / span
    return focal_rows * tan_top, focal_rows
