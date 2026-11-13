"""The lens: an equidistant projection of three shared numbers, and the delivered frame's scale."""

import math


def degrees_per_pixel(fov_h: float, mode_width: int) -> float:
    """The angle (°) of one output column; `mode_width` is the un-cropped width `fov_h` is quoted for."""
    if mode_width <= 0:
        return 0.0
    return fov_h / float(mode_width)


def output_focal(fov_h: float, mode_width: int) -> float:
    """The delivered frame's focal length (px/rad), the same for its columns and its rows at the horizon."""
    dpp: float = degrees_per_pixel(fov_h, mode_width)
    return 1.0 / math.radians(dpp) if dpp > 0.0 else 0.0


def source_lens(src_size: tuple[int, int], mode_width: int, fov_h: float,
                lens_fov: float = 0.0,
                lens_center: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float, float]:
    """The lens in source pixels: (focal px/rad, cx, cy); `lens_fov` 0 takes `fov_h`."""
    src_w, src_h = src_size
    field: float = lens_fov if lens_fov > 0.0 else fov_h
    focal: float = mode_width / math.radians(field) if field > 0.0 and mode_width > 0 else 0.0
    # A frame of width W spans 0..W-1, so its center sits at (W-1)/2: that half pixel keeps the center on axis.
    return focal, (src_w - 1) / 2.0 + lens_center[0], (src_h - 1) / 2.0 + lens_center[1]


def lens_field(focal: float, width: int) -> float:
    """The field (°) an equidistant lens of `focal` px/rad spans across `width` px."""
    return math.degrees(width / focal) if focal > 0.0 else float('nan')


def rectilinear_field(focal: float, cx: float, width: int) -> float:
    """The field (°) an undistorted frame of `focal` px and optical center `cx` spans across `width` px."""
    if focal <= 0.0:
        return float('nan')
    return math.degrees(math.atan((cx + 0.5) / focal) + math.atan((width - cx - 0.5) / focal))


def lens_deviation(focal: float, cx: float, cy: float, src_size: tuple[int, int],
                   mode_width: int, fov_h: float, lens_fov: float = 0.0,
                   lens_center: tuple[float, float] = (0.0, 0.0)) -> float:
    """The largest bearing error (°) a unit of this calibration has under the shared lens: its center offset
    or its focal mismatch at the frame's edge."""
    shared_focal, shared_cx, shared_cy = source_lens(src_size, mode_width, fov_h, lens_fov, lens_center)
    if shared_focal <= 0.0 or focal <= 0.0:
        return float('nan')
    center: float = math.hypot(cx - shared_cx, cy - shared_cy) / shared_focal
    edge: float = (mode_width / 2.0) * abs(1.0 / shared_focal - 1.0 / focal)
    return math.degrees(max(center, edge))
