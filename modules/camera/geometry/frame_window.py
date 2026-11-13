"""The delivered frame's rows as elevations: cylindrical, pinned at the sensor's lowest reach."""

import math
from dataclasses import dataclass

from .lens import degrees_per_pixel, output_focal, source_lens


@dataclass(frozen=True)
class FrameWindow:
    """A frame's rows as elevations: `horizon_px - row = focal * tan(elevation)`. Elevations in degrees,
    `horizon_px` the un-flipped row of elevation 0, `focal` in px/rad."""
    elevation_bottom: float
    elevation_top: float
    horizon_px: float
    focal: float

    def elevation(self, row: float) -> float:
        """Elevation (°) of an un-flipped output row."""
        if self.focal <= 0.0:
            return 0.0
        return math.degrees(math.atan((self.horizon_px - row) / self.focal))

    def row(self, elevation: float) -> float:
        """Un-flipped output row of an elevation (°)."""
        return self.horizon_px - self.focal * math.tan(math.radians(elevation))


def frame_window(src_size: tuple[int, int], out_size: tuple[int, int], mode_width: int,
                 fov_h: float, tilt: float, lens_fov: float = 0.0,
                 lens_center: tuple[float, float] = (0.0, 0.0)) -> FrameWindow:
    """The `FrameWindow` a configuration delivers: the same the warp makes."""
    out_h: int = out_size[1]
    src_h: int = src_size[1]
    focal: float = output_focal(fov_h, mode_width)
    lens_focal, _, cy = source_lens(src_size, mode_width, fov_h, lens_fov, lens_center)
    if focal <= 0.0 or lens_focal <= 0.0:
        return FrameWindow(0.0, 0.0, (out_h - 1) / 2.0, 0.0)
    reach_below: float = math.degrees((src_h - 1 - cy) / lens_focal)
    bottom: float = tilt - reach_below
    horizon: float = (out_h - 1) + focal * math.tan(math.radians(bottom))
    top: float = math.degrees(math.atan(horizon / focal))
    return FrameWindow(bottom, top, horizon, focal)


def horizon_row(window: FrameWindow, out_h: int) -> float:
    """The horizon's row in a delivered frame, 0 top to 1 bottom; may fall outside [0, 1]."""
    return window.horizon_px / (out_h - 1) if out_h > 1 else 0.5


def frame_fov(fov_h: float, mode_width: int, out_size: tuple[int, int],
              window: FrameWindow) -> tuple[float, float]:
    """The horizontal and vertical field (°) a delivered frame covers."""
    dpp: float = degrees_per_pixel(fov_h, mode_width)
    return dpp * out_size[0], window.elevation_top - window.elevation_bottom
