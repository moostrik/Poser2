"""One observation turned into one mark on the strip. No GL, no board, no settings — arithmetic.

A mark is everything drawn for one observation: its **line**, its **foot tick**, its **field** and
its **label**. Built once by the compositor for both `MarkRenderer` (line, tick, field) and
`LabelRenderer` (label), so the two cannot disagree about where a person is.

**A mark is the tracker's belief, and its two axes use two different depths on purpose.**

- **x** is the view's own `world_angle`, which the tracker derives at the fixed
  `rig.parallax_radius`. What the tracker emits is a blend of a person's active views
  (`Seams.world_azimuth`); the show's azimuth is drawn by the azimuth overlay. The field goes through the same depth, since
  it is a statement about the same azimuth. So a mark sits a small constant distance from its own
  pixels (the image is stitched at `focus_radius`), and two views of one person at a seam differ by
  the residual for their depth: a depth indicator, not an error.
- **y** goes through the person's own `annotation.distance`. There the lens height cancels,
  `atan(tan(-atan(h / d)) * d / R) = atan(-h / R)`, which is the grid's zone-band formula: **the
  foot tick sits on the R 3.5 edge iff the tracker reports this person at R 3.5.** On the parallax
  cylinder that exactness would be lost. The distance is unclamped, so a person past the far edge
  shows as a tick above the zone band.

A mark's **field** is the join range that governs it, as a width (`_field`). A **rejected**
detection is a grey line with no field, labelled with its rejection (`rejection_label`). A **LOST**
mark keeps its line, fading to grey over `lost_timeout` while its field fades out (`_colors`).
The primary view of a person is drawn strong; the other views are **passive**.
"""

# Standard library imports
import math
from dataclasses import dataclass

# Local application imports
from modules.tracker import PanoramicAnnotation, Rejection, Tracklet, TrackingStatus, camera_azimuth, \
    camera_local_to_azimuth, centre_distance, elevation_from_row, focus_distance, wrap180

from .strip import centre_elevation, strip_y

Color = tuple[float, float, float, float]

# How lit a passive view's line is, against the primary's 1.
_PASSIVE_ALPHA: float = 0.8


@dataclass(frozen=True)
class MarkContext:
    """Everything a mark needs from the tracker's published numbers, rebuilt by the compositor each
    tick and passed in so this module stays pure.

    `row_model` is the frames' (horizon_row, focal_rows) (`projection.row_model`); `elevation_window` the
    strip's (top, bottom) at the rig centre; `now` and `lost_timeout` time a LOST mark's fade.
    """
    cam_fov: float
    target_fov: float
    camera_radius: float
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
    """Where one observation lands on the strip, and what its label says."""
    world_id: int
    cam_id: int
    x: float                                        # normalised strip x, from the world azimuth
    top_y: float                                    # normalised strip y of the box's top row
    bottom_y: float                                 # ... and of the feet
    has_foot: bool                                  # whether `bottom_y` is a floor reading at all
    field_x: float                                  # left edge of the field...
    field_w: float                                  # ... and its width; both normalised strip x
    color: tuple[float, float, float, float]        # the line's colour, alpha carrying confidence
    field_color: tuple[float, float, float, float]  # the field's world colour; alpha = how visible
    field_outline: bool                             # a passive or LOST view: outlined, not filled
    is_primary: bool
    rejected: bool                                  # a detection a filter rejected: no world, grey
    label: str


def build_marks(observations: list[Tracklet], primaries: set[int],
                colors: list[tuple[float, float, float, float]],
                rejected_color: tuple[float, float, float, float],
                context: MarkContext) -> list[Mark]:
    """A mark per usable observation. Rejected detections first, under everyone; primaries last, so
    they are drawn over their passive views."""
    marks: list[Mark] = []
    for tracklet in observations:
        mark: Mark | None = _mark(tracklet, primaries, colors, rejected_color, context)
        if mark is not None:
            marks.append(mark)
    marks.sort(key=lambda m: (not m.rejected, m.is_primary))
    return marks


def _mark(tracklet: Tracklet, primaries: set[int],
          colors: list[tuple[float, float, float, float]],
          rejected_color: tuple[float, float, float, float],
          c: MarkContext) -> Mark | None:
    if tracklet is None or tracklet.is_removed:
        return None
    if not isinstance(tracklet.annotation, PanoramicAnnotation):
        return None

    annotation: PanoramicAnnotation = tracklet.annotation
    roi_bottom: float = tracklet.roi.y + tracklet.roi.height
    # The rows go through the person's own distance (module docstring); `centre_dist` is the radius
    # the label prints as `R` and the foot tick is drawn at.
    has_foot: bool = math.isfinite(annotation.distance)
    if has_foot:
        cam_distance: float = max(1e-6, annotation.distance)
        centre_dist: float = centre_distance(annotation.local_angle - c.cam_fov / 2.0,
                                             cam_distance, c.camera_radius)
        top_y: float = _row_y(tracklet.roi.y, c, cam_distance, centre_dist)
        bottom_y: float = _foot_y(c, centre_dist)
    else:
        # Feet at or above the horizon: no floor distance. Rows go through the parallax cylinder,
        # where the x already is, and there is no foot tick.
        phi: float = wrap180(annotation.world_angle - camera_azimuth(tracklet.cam_id, c.target_fov))
        cylinder: float = focus_distance(phi, c.camera_radius, c.parallax_radius)
        centre_dist = math.inf
        top_y = _row_y(tracklet.roi.y, c, cylinder, c.parallax_radius)
        bottom_y = _row_y(roi_bottom, c, cylinder, c.parallax_radius)

    # Rejected by a filter and never given a world: a grey line, no field, labelled with why.
    rejected: bool = annotation.rejected is not None and tracklet.id < 0
    is_primary: bool = tracklet.obs_id in primaries
    if rejected:
        field_lo = field_hi = annotation.world_angle
        color: tuple[float, float, float, float] = rejected_color
        field_color: tuple[float, float, float, float] = (*rejected_color[:3], 0.0)
    else:
        field_lo, field_hi = _field(annotation, tracklet.cam_id, c)
        color, field_color = _colors(tracklet, is_primary, colors, rejected_color, c)

    return Mark(
        world_id=tracklet.id,
        cam_id=tracklet.cam_id,
        x=(annotation.world_angle % 360.0) / 360.0,
        top_y=top_y,
        bottom_y=bottom_y,
        has_foot=has_foot,
        field_x=(field_lo % 360.0) / 360.0,
        field_w=((field_hi - field_lo) % 360.0) / 360.0,
        color=color,
        field_color=field_color,
        field_outline=not is_primary or tracklet.status == TrackingStatus.LOST,
        is_primary=is_primary,
        rejected=rejected,
        label=_label(tracklet, annotation, centre_dist, rejected, c),
    )


def _fade(tracklet: Tracklet, c: MarkContext) -> float:
    """How far a LOST observation is toward being forgotten: 0 just lost, 1 at `lost_timeout`.
    0 for anything the device is still detecting."""
    if tracklet.status != TrackingStatus.LOST:
        return 0.0
    return min(1.0, max(0.0, c.now - tracklet.last_active) / max(1e-6, c.lost_timeout))


def _colors(tracklet: Tracklet, is_primary: bool, colors: list[Color], grey: Color,
            c: MarkContext) -> tuple[Color, Color]:
    """(line, field) colours: the world colour, alpha saying how much to trust it.

    A passive view has its line dimmed; its field is told apart by shape (`Mark.field_outline`). A
    **LOST** observation is still an identity the tracker holds, so it keeps its mark, dimmed, and
    over `lost_timeout` its line fades to grey and its field fades out — fully grey is when a
    rejected detection's grey line would take over.
    """
    r, gr, b, a = colors[tracklet.id % len(colors)] if colors else (1.0, 1.0, 1.0, 1.0)
    if tracklet.status == TrackingStatus.LOST:
        t: float = _fade(tracklet, c)
        line: Color = (r + (grey[0] - r) * t, gr + (grey[1] - gr) * t, b + (grey[2] - b) * t,
                       a * _PASSIVE_ALPHA)
        return line, (r, gr, b, 1.0 - t)
    return (r, gr, b, a if is_primary else a * _PASSIVE_ALPHA), (r, gr, b, 1.0)


def rejection_label(reason: Rejection, zone_max_radius: float) -> str:
    """The label text for a rejection: the whole label of a rejected detection, and the suffix on a
    tracked person's label when a filter stopped counting them."""
    if reason == Rejection.PAST_EDGE:
        return f'past R{zone_max_radius:g}'
    return {Rejection.YOUNG: 'young', Rejection.SMALL: 'small', Rejection.DEAD_ZONE: 'dead zone',
            Rejection.NO_ID: 'no id'}[reason]


def _label(tracklet: Tracklet, annotation: PanoramicAnnotation, centre_dist: float,
           rejected: bool, c: MarkContext) -> str:
    """The label beside a mark: the rejection for a rejected detection, the readout otherwise,
    followed by the rejection for a tracked person a filter stopped counting.

    Fixed width, so the right-edge flip threshold cannot wobble as the digits change. `R` is the
    radius the foot tick is drawn at. `R` and `H` read low until `track.foot_offset` is calibrated.
    """
    reason: str = '' if annotation.rejected is None else rejection_label(annotation.rejected, c.zone_max_radius)
    if rejected:
        return reason
    distance: str = f'R{centre_dist:.1f}m' if math.isfinite(centre_dist) else 'R-'
    readout: str = (f'#{tracklet.id} c{tracklet.cam_id} '
                    f'az{annotation.world_angle % 360.0:03.0f} '
                    f'{distance} '
                    f'H{annotation.height:.1f}m')
    return f'{readout} {reason}' if reason else readout


def _field(annotation: PanoramicAnnotation, cam_id: int, c: MarkContext) -> tuple[float, float]:
    """(low, high) azimuth of the field: the rule that decides what this observation may be joined to.

    **A pair test**: both rules are `|Δ| ≤ angle`, so fields `angle` wide touch exactly when the
    difference equals the angle — two fields of one colour that overlap are two observations the
    tracker will join.

    Inside the overlap the seam rule is drawn (`seam.link_angle`, world azimuth); outside it the only
    rule is the same camera re-finding a person (`reacquire_angle`, local angle). Re-acquisition also
    applies inside the overlap, but the seam rule is the one being tuned there.
    """
    if annotation.overlap:
        # Symmetric about the mark, unclamped: its partner is in the neighbouring camera's field of view.
        half: float = c.link_angle / 2.0
        centre: float = annotation.world_angle
        return (centre - half, centre + half)

    # A local-angle rule, carried into azimuth through the tracker's own map (`parallax_radius`), so
    # positions and widths take the same map and the overlap still answers the rule exactly — and
    # the field reads narrower than `reacquire_angle` on the degree grid. Clamped to the camera's own
    # field of view, which cannot hide a real pair: both centres lie inside it.
    half_local: float = c.reacquire_angle / 2.0
    local: float = annotation.local_angle
    lo: float = camera_local_to_azimuth(max(0.0, local - half_local), cam_id, c.cam_fov,
                                        c.target_fov, c.camera_radius, c.parallax_radius)
    hi: float = camera_local_to_azimuth(min(c.cam_fov, local + half_local), cam_id, c.cam_fov,
                                        c.target_fov, c.camera_radius, c.parallax_radius)
    return (lo, hi)


def _row_y(row: float, c: MarkContext, cam_distance: float, centre_dist: float) -> float:
    """A normalised frame row to a normalised strip y, converted to the rig centre through the given
    distances. Not clamped — an extrapolated box may run past the frame — the renderer clips."""
    cam_elevation: float = elevation_from_row(row, *c.row_model)
    return strip_y(centre_elevation(cam_elevation, cam_distance, centre_dist), c.elevation_window)


def _foot_y(c: MarkContext, centre_dist: float) -> float:
    """The strip y of the floor at radius `centre_dist` — the foot tick.

    The same number as `_row_y` of the foot row through the person's own distance (the lens height
    cancels), written as `GridRenderer._zone_band`'s own formula so the tick and the zone edges
    cannot drift apart. Below the strip for a box extrapolated far below the frame; not drawn then.
    """
    elevation: float = -math.degrees(math.atan(c.camera_height / max(1e-6, centre_dist)))
    return strip_y(elevation, c.elevation_window)
