# Standard library imports
import math

# Third-party imports
from OpenGL.GL import * # type: ignore

# Local application imports
from modules.gl import Text
from modules.tracker import PanoramicTrackerSettings, camera_azimuth

from ...shaders import DrawColoredRectangle
from ..LayerBase import LayerBase
from .settings import PanoramaLayerSettings, \
    AXIS_COLOR, GRID_COLOR, HORIZON_COLOR, HORIZON_PX, LABEL_BG, LABEL_FG, \
    OVERLAP_COLOR, SEAM_COLOR, ZONE_COLOR
from .strip import strip_y


class GridRenderer(LayerBase):
    """Every reference line measured in the strip's own two axes — centre azimuth and centre
    elevation — against its degree labels. Anything defined on a camera's own frame has no single
    azimuth and lives in `SeamRenderer`, against the picture.

    **In azimuth**, exact and depth-free:

    - the **degree lattice** at `grid_degrees`, faint;
    - the **sector boundaries** in orange at `target_fov · cam_id`;
    - the **camera axes** in blue at `camera_azimuth`, where the parallax correction is the identity;
    - the **overlap** in yellow, two verticals per seam at `± rig.overlap / 2`: exactly where a
      mark's field changes width (`Rig._update_overlap_band`). Not where the pictures meet — the
      flag is deliberately generous.

    **In elevation**, the horizon and the **zone band**: the tracked floor, a translucent band between
    the two radii at `atan(camera_height / R)` below the horizon, the same at every azimuth. The only
    thing on the strip in metres, so the one a tape on the floor checks, and what a mark's foot tick is
    read against. A band rather than two lines, because at some presets the zone runs below the
    strip's bottom and a fill running off the edge says so, where a line pinned there would claim a
    wrong elevation.
    """

    def __init__(self, num_cams: int, tracker: PanoramicTrackerSettings,
                 settings: PanoramaLayerSettings) -> None:
        self._num_cams: int = max(1, num_cams)
        self._tracker: PanoramicTrackerSettings = tracker
        self._settings: PanoramaLayerSettings = settings
        self._rect: DrawColoredRectangle = DrawColoredRectangle()
        self._text: Text = Text()

        self._width: int = 1
        self._height: int = 1
        self._elevation_window: tuple[float, float] = (0.0, 0.0)

    @property
    def _target_fov(self) -> float:
        """The sector one camera owns — the tracker's own `360 / num_cameras`."""
        return 360.0 / self._num_cams

    def set_geometry(self, elevation_window: tuple[float, float]) -> None:
        self._elevation_window = elevation_window

    def allocate(self, width: int, height: int, internal_format: int) -> None:
        self._rect.allocate()
        self._text.allocate()
        self._width = max(1, width)
        self._height = max(1, height)

    def deallocate(self) -> None:
        self._rect.deallocate()
        self._text.deallocate()

    def update(self) -> None:
        pass

    def draw(self) -> None:
        spacing: float = max(1.0, self._settings.grid_degrees)
        px_x: float = 1.0 / self._width
        px_y: float = 1.0 / self._height

        # The one fill here, and first, so every line below stays legible over it.
        self._zone_band()

        azimuth: float = 0.0
        while azimuth < 360.0:
            self._vertical(azimuth, px_x, GRID_COLOR)
            azimuth += spacing

        top, bottom = self._elevation_window
        elevation: float = spacing
        while elevation < max(abs(top), abs(bottom)):
            if elevation < top:
                self._horizontal(elevation, px_y, GRID_COLOR)
            if -elevation > bottom:
                self._horizontal(-elevation, px_y, GRID_COLOR)
            elevation += spacing

        self._reference_verticals(px_x)

        # The horizon last, so no grid line is drawn over it. It is the reference the levelling
        # check is read against (tape at lens height must sit on it), so it has to be unmistakable
        # rather than one more line.
        self._horizontal(0.0, HORIZON_PX * px_y, HORIZON_COLOR)

        self._draw_labels(spacing)

    def _reference_verticals(self, px_x: float) -> None:
        """The rig's own bearings: sector boundaries, camera axes, overlap edges.

        All three are azimuth and nothing else — no camera frame, no depth, no
        `camera_local_to_azimuth` — which is exactly what lets them be read straight off the degree
        labels, and why they sit in the lattice rather than beside the dead zone.
        """
        half: float = self._tracker.rig.overlap / 2.0
        for cam_id in range(self._num_cams):
            seam: float = self._target_fov * cam_id
            if half > 0.0:                    # 0 once the sectors stop meeting: nothing to mark
                self._vertical(seam - half, px_x, OVERLAP_COLOR)
                self._vertical(seam + half, px_x, OVERLAP_COLOR)
            self._vertical(camera_azimuth(cam_id, self._target_fov), 2.0 * px_x, AXIS_COLOR)
            self._vertical(seam, 2.0 * px_x, SEAM_COLOR)

    def _zone_band(self) -> None:
        """The tracked floor, as the band of elevations it subtends at the rig centre.

        A floor circle of radius `R` sits `atan(camera_height / R)` below the horizon, so the zone
        between two radii is a band of rows — the same at every azimuth, which is what makes it
        straight-sided rather than a curve. The far edge is the higher row, since a more distant
        floor is nearer the horizon.

        Clipped to the strip's window, and the clipping carries meaning: at P720 / tilt 15 the near
        edge is 1.3° below the window bottom, so the band simply runs off the bottom of the strip
        and says "the tracked floor continues past here". A whole zone below the window draws
        nothing at all, which is then the honest answer.
        """
        rig = self._tracker.rig
        top, bottom = self._elevation_window

        def row(radius: float) -> float:
            elevation: float = -math.degrees(
                math.atan(rig.camera_height / max(1e-6, radius)))
            return strip_y(min(max(elevation, bottom), top), self._elevation_window)

        y_far: float = row(rig.zone_max_radius)
        y_near: float = row(rig.zone_min_radius)
        if y_near - y_far <= 0.0:
            return
        self._rect.use(0.0, y_far, 1.0, y_near - y_far, *ZONE_COLOR)

    def _vertical(self, azimuth: float, width: float,
                  color: tuple[float, float, float, float]) -> None:
        self._rect.use((azimuth % 360.0) / 360.0, 0.0, width, 1.0, *color)

    def _horizontal(self, elevation: float, height: float,
                    color: tuple[float, float, float, float]) -> None:
        # Rect y is top-down, and the window's top elevation is the strip's top row. The horizon is
        # only at mid-height when the window is symmetric, which a tilted camera's is not. Centred
        # on the elevation, so a thicker line does not drift below the row it marks.
        self._rect.use(0.0, self._elevation_y(elevation) - height / 2.0, 1.0, height, *color)

    def _elevation_y(self, elevation: float) -> float:
        return strip_y(elevation, self._elevation_window)

    def _draw_labels(self, spacing: float) -> None:
        """Degree labels along the top, the horizon named, and the geometry in the corner.

        Labelled every other grid line when the spacing is fine, so they never collide.
        """
        stride: float = spacing if spacing >= 15.0 else spacing * 2.0
        azimuth: float = 0.0
        while azimuth < 360.0:
            x: float = (azimuth / 360.0) * self._width + 3
            self._text.draw_box_text(x, 3, f'{azimuth:.0f}', LABEL_FG, LABEL_BG,
                                     self._width, self._height)
            azimuth += stride

        horizon_px: float = self._elevation_y(0.0) * self._height
        self._text.draw_box_text(3, max(3.0, horizon_px - 24), 'horizon', HORIZON_COLOR, LABEL_BG,
                                 self._width, self._height)

        # One line: the strip's own geometry, then the seam settings in the units they are
        # drawn in, so a band's or a mark field's width can be read off the strip and checked against
        # the number that produced it.
        #
        # `elev` is the STRIP's window — at the rig centre, at `focus_radius` — not the frame's.
        # It reads narrower than `rig.angle_bottom/top` (−17..47 against −21..52 here) because
        # `elevation_window` converts the camera's band to the centre's view and takes the ratio at
        # its tightest bearing, straight ahead, so no column of the strip fades to black. It
        # therefore moves with `focus_radius` while the rig's numbers do not.
        top, bottom = self._elevation_window
        seam = self._tracker.seam
        rig = self._tracker.rig
        # Radii throughout, so the zone here, a mark's `R` and a tape from the fixture are one
        # number — `zone 1.5..3.5` and a person labelled `R3.5` are saying the same thing.
        footer: str = (f'R{self._settings.focus_radius:.2f}m  fov {self._tracker.fov:.0f}  '
                       f'tilt {self._settings.tilt:.0f}  elev {bottom:.0f}..{top:.0f}  '
                       f'zone {rig.zone_min_radius:.2f}..{rig.zone_max_radius:.2f}m  '
                       f'overlap {rig.overlap:.1f}°  dead {seam.dead_zone:.1f}°  '
                       f'link {seam.link_angle:.1f}°/{seam.link_height:.2f}  '
                       f'reacquire {self._tracker.reacquire_angle:.1f}°')
        self._text.draw_box_text(3, self._height - 22, footer, LABEL_FG, LABEL_BG,
                                 self._width, self._height)
