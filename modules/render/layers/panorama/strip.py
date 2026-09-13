"""The 360-degree strip's own geometry: its rows, its extent, and the 0/360 join.

x is centre azimuth, linear in degrees; y is the TANGENT of centre elevation, like the camera frames'
rows. The projection between a camera's frame and the world — azimuth to column and back — is the
tracker's (`modules/tracker/panoramic/projection.py`), since the tracker's azimuths and the stitch's
placement must be one arithmetic. What is here only the display needs. `strip_y`, `camera_elevation` and `panorama_coverage` are transcribed
into `panoramicstitch.frag`; keep them in step.

`elevation_window` is the one function that names its depth, `focus_radius`: it sets the strip's
*single* y scale, at the picture's depth, for everything drawn on it.
"""

import math

from modules.tracker import azimuth_to_camera_x, focus_distance


def strip_spans(x: float, width: float) -> list[tuple[float, float]]:
    """Normalised x spans of a band `width` wide with its LEFT edge at `x`, wrapped at the join.

    The strip's left and right edges are the same azimuth, so a band that runs off one comes back
    on the other and has to be drawn as two quads. One function because four things need it — the
    dead-zone bands, the mark fields and the foot tick — and a band silently clipped at azimuth
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


def camera_elevation(elevation: float, bearing: float, camera_radius: float,
                     depth_radius: float) -> float:
    """The elevation (degrees) a camera sees for a point the rig centre sees at `elevation`.

    The vertical half of the same triangle `azimuth_to_camera_x` solves horizontally, and it has
    to be solved too: a camera `camera_radius` out from the centre is *closer* to the near wall of
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
    distance: float = focus_distance(bearing, camera_radius, depth_radius)
    if distance <= 1e-9:
        return elevation
    return math.degrees(math.atan(math.tan(math.radians(elevation)) * depth_radius / distance))


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


def elevation_window(band: tuple[float, float], camera_radius: float,
                     focus_radius: float) -> tuple[float, float]:
    """(top, bottom) elevation of the 360-degree strip, measured AT THE RIG CENTRE.

    `band` is the (bottom, top) the delivered frames carry, measured at the camera — the tracker's
    published `angle_bottom` and `angle_top`. Converted to the centre's point of view, at the
    bearing where the conversion is tightest. `tan(e_centre) = tan(e_cam) * d / focus_radius`, and
    `d` is smallest straight ahead (`focus_radius - camera_radius`), so taking the window there
    guarantees every column of the strip is filled rather than fading to black near the camera axes.

    **`focus_radius`, not `depth_radius`** — the one function here that names its depth, because it
    has one caller and one meaning: it fixes the strip's **single, shared** y scale at the
    picture's depth, and everything drawn on the strip is then placed on that same ruler. So it
    moves the whole strip together and can never break a comparison between two things on it: drag
    `focus_radius` and the horizon stays at y 0.7791, because the scale and the thing measured
    move as one. That is the opposite of the per-call depths in `projection.py`, which do break
    comparisons.
    """
    radius: float = max(1e-6, focus_radius)
    ratio: float = max(0.0, radius - camera_radius) / radius
    low, high = band
    return (math.degrees(math.atan(math.tan(math.radians(high)) * ratio)),
            math.degrees(math.atan(math.tan(math.radians(low)) * ratio)))


def panorama_coverage(azimuth: float, num_cameras: int, cam_fov: float, target_fov: float,
                      camera_radius: float, depth_radius: float) -> int:
    """How many cameras see this azimuth — the divisor an averaging blend needs.

    At `camera_radius = 0` this is 2 within `(cam_fov - target_fov) / 2` of every seam and 1
    elsewhere. With the parallax correction on, the bands are narrower (a camera pushed outward
    covers less of the cylinder as measured from the centre) but the shape is the same, and it is
    never 0 as long as the cameras' fields sum past 360.
    """
    return sum(
        1 for cam_id in range(num_cameras)
        if azimuth_to_camera_x(azimuth, cam_id, cam_fov, target_fov,
                               camera_radius, depth_radius) is not None
    )
