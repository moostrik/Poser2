"""How a rig camera is mounted and what its lens is, and what its open log says of both."""

from dataclasses import dataclass

from ..settings import RigCameraReadings
from .lens import lens_deviation, lens_field
from .warp import coverage_summary, frame_coverage
from .frame_window import frame_window


@dataclass
class MountConfig:
    """How a camera is mounted and what its lens is, as the warp and the correction need it."""
    flip_h: bool
    flip_v: bool
    fov_h: float                # ° the delivered frame spans, quoted for the un-cropped sensor width
    tilt: float = 0.0           # °, positive aimed up; exclusive with `keystone` (the OAK-D's warp)
    keystone: float = 0.0       # fraction of the frame (the OAK-D's warp)
    lens_fov: float = 0.0       # the lens (`source_lens`); 0 takes `fov_h`
    lens_center_x: float = 0.0
    lens_center_y: float = 0.0

    @property
    def lens_center(self) -> tuple[float, float]:
        return (self.lens_center_x, self.lens_center_y)


def frame_summary(src_size: tuple[int, int], out_size: tuple[int, int], mount: MountConfig) -> str:
    """One line for the open log: the window the warp delivers at the mount's tilt and the rows the sensor covers."""
    window = frame_window(src_size, out_size, src_size[0], mount.fov_h, mount.tilt, mount.lens_fov, mount.lens_center)
    coverage = frame_coverage(src_size, out_size, src_size[0], mount.fov_h, mount.tilt, mount.flip_h, mount.flip_v,
                              mount.lens_fov, mount.lens_center)
    return (f'frame {out_size[0]}x{out_size[1]}: elevation {window.elevation_bottom:+.1f} to '
            f'{window.elevation_top:+.1f} deg, horizon at row {window.horizon_px:.1f}, {window.focal:.1f} px/rad; '
            f'{coverage_summary(coverage, out_size, src_size[0], mount.fov_h, mount.flip_h)}')


def lens_report(readings: RigCameraReadings, lens: tuple[float, float, float], src_size: tuple[int, int],
                mount: MountConfig) -> str:
    """A unit's own lens, (focal px/rad, cx, cy) at `src_size`, against the shared one: its field and its largest
    bearing error into the readings (`fov_factory`, `lens_error`), and one line for the open log."""
    focal, cx, cy = lens
    field: float = lens_field(focal, src_size[0])
    error: float = lens_deviation(focal, cx, cy, src_size, src_size[0], mount.fov_h, mount.lens_fov, mount.lens_center)
    readings.fov_factory = field
    readings.lens_error = error
    return (f'lens: field {field:.1f} deg across {src_size[0]} px, center offset '
            f'({cx - (src_size[0] - 1) / 2.0:+.1f}, {cy - (src_size[1] - 1) / 2.0:+.1f}) px; '
            f'{error:.2f} deg off the shared lens')
