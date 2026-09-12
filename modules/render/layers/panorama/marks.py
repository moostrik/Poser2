"""One observation turned into one mark on the strip. No GL, no board, no settings — arithmetic.

Kept out of the renderers because two of them draw the same marks: `ObservationRenderer` draws the
lines, `LabelRenderer` the text beside them, and if each worked the geometry out for itself the two
could disagree about where a person is. The compositor builds the list once and hands it to both.

**Which model a mark uses, and why it matters.** A mark is placed where the *tracker* believes the
person is: its azimuth is the fused `world_angle`, and its elevations are re-projected to the rig
centre through that person's own estimated distance (`centre_elevation`). The image underneath is
placed where the *focus cylinder* says (`camera_elevation`). So a mark that sits above or below its
own pixels is not a drawing error — it is the distance model disagreeing with the picture, which is
the open question in CALIBRATION.md, now visible per person and in place.
"""

# Standard library imports
import math
from dataclasses import dataclass

# Local application imports
from modules.tracker import PanoramicAnnotation, Tracklet, TrackingStatus, \
    centre_distance, centre_elevation


@dataclass(frozen=True)
class Mark:
    """Where one observation lands on the strip, and what to say about it."""
    obs_id: int
    world_id: int
    cam_id: int
    x: float                                        # normalised strip x, from the world azimuth
    top_y: float                                    # normalised strip y of the box's top row
    bottom_y: float                                 # ... and of its bottom row: the feet
    color: tuple[float, float, float, float]        # the world colour, alpha carrying confidence
    is_primary: bool
    label: str


def elevation_y(elevation: float, elevation_window: tuple[float, float]) -> float:
    """Normalised, top-down y of an elevation in the strip."""
    top, bottom = elevation_window
    return (top - elevation) / max(1e-6, top - bottom)


def build_marks(observations: list[Tracklet], primaries: set[int],
                colors: list[tuple[float, float, float, float]],
                cam_fov: float, vfov: float, ring_radius: float,
                elevation_window: tuple[float, float]) -> list[Mark]:
    """A mark per usable observation, primaries last so they are drawn over their candidates."""
    marks: list[Mark] = []
    for tracklet in observations:
        mark: Mark | None = _mark(tracklet, primaries, colors, cam_fov, vfov, ring_radius,
                                  elevation_window)
        if mark is not None:
            marks.append(mark)
    marks.sort(key=lambda m: m.is_primary)
    return marks


def _mark(tracklet: Tracklet, primaries: set[int],
          colors: list[tuple[float, float, float, float]],
          cam_fov: float, vfov: float, ring_radius: float,
          elevation_window: tuple[float, float]) -> Mark | None:
    if tracklet is None or tracklet.is_removed:
        return None
    if not isinstance(tracklet.annotation, PanoramicAnnotation):
        return None

    annotation: PanoramicAnnotation = tracklet.annotation
    # The bearing off this camera's own axis, which is what the parallax triangle takes.
    bearing: float = annotation.local_angle - cam_fov / 2.0
    cam_distance: float = max(1e-6, annotation.distance)
    centre_dist: float = centre_distance(bearing, cam_distance, ring_radius)

    top_y: float = _row_y(tracklet.roi.y, vfov, cam_distance, centre_dist, elevation_window)
    bottom_y: float = _row_y(tracklet.roi.y + tracklet.roi.height, vfov, cam_distance,
                             centre_dist, elevation_window)

    is_primary: bool = tracklet.obs_id in primaries
    r, g, b, a = colors[tracklet.id % len(colors)] if colors else (1.0, 1.0, 1.0, 1.0)
    # The colour says who; the alpha says how much to trust it. A LOST observation is still
    # anchoring a seam crossing, so it is drawn, but faintly; a loser at a seam is dimmed so the
    # primary reads as the one in charge.
    if tracklet.status == TrackingStatus.LOST:
        a *= 0.25
    elif not is_primary:
        a *= 0.5

    return Mark(
        obs_id=tracklet.obs_id,
        world_id=tracklet.id,
        cam_id=tracklet.cam_id,
        x=(annotation.world_angle % 360.0) / 360.0,
        top_y=top_y,
        bottom_y=bottom_y,
        color=(r, g, b, a),
        is_primary=is_primary,
        # Fixed width, so the right-edge flip threshold is the same for everybody and cannot wobble
        # as the digits change. `R` is the drafting radius: this distance is from the rig centre,
        # where the footer's `Ø` is a diameter — the two differ by a factor of two and must not be
        # read as the same kind of number.
        label=f'#{tracklet.id} c{tracklet.cam_id} '
              f'az{annotation.world_angle % 360.0:03.0f} R{centre_dist:.1f}m',
    )


def _row_y(row: float, vfov: float, cam_distance: float, centre_dist: float,
           elevation_window: tuple[float, float]) -> float:
    """A normalised frame row to a normalised strip y.

    The delivered frame is equirectangular and levelled, so a row *is* an elevation measured at the
    camera: `e_cam = (0.5 - row) * vfov`, positive upward. A row may legitimately fall outside
    [0, 1] — the device tracker extrapolates a partly visible person, and that is real information
    about how close they are — so nothing is clamped here; the renderer clips when it draws.
    """
    cam_elevation: float = (0.5 - row) * vfov
    return elevation_y(centre_elevation(cam_elevation, cam_distance, centre_dist),
                       elevation_window)
