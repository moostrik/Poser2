"""Compare how the cameras are mounted against how the preset says they are."""

import math

from .settings import CameraSettings, MountCheckSettings


def mount_deviation(camera: CameraSettings) -> tuple[float, float]:
    """The camera's signed ``(tilt, roll)`` deviation in degrees; NaN where it has no reading.

    - **tilt** — ``tilt_measured - tilt``. `tilt` is the value the warp mesh was baked with, so
      a camera that does not sit at it is being un-tilted by the wrong amount.
    - **roll** — ``roll_measured``. There is no configured roll: `warp_mesh_points` has no
      roll term at all, so the only correct value is zero, and any roll tips the horizon in a way
      the panorama cannot distinguish from a wrong tilt.
    """
    return (camera.readings.tilt_measured - camera.tilt, camera.readings.roll_measured)


class MountCheck:
    """Turns the per-camera IMU readings into one verdict a person will actually read.

    Both deviations of every camera (`mount_deviation`) are compared against the same
    `tolerance`, and `mount` is True only when every reported value is within it. Which camera and
    axis is off is not repeated here: the renderer draws each camera's numbers on its own view.

    `fov_factory` and `lens_error` are deliberately *not* checked. They describe the unit's lens
    against the shared one, which is a property of the build, not of how the tripod was set
    this morning — alarming on them would mean a warning every launch, which is the fastest way
    to teach someone to ignore warnings.
    """

    def __init__(self, cameras: list[CameraSettings], settings: MountCheckSettings) -> None:
        self._cameras: list[CameraSettings] = cameras
        self._settings: MountCheckSettings = settings

    def update(self) -> None:
        """Recompute the verdict. Cheap; call it on the render tick."""
        # Cameras that never reported are left out rather than counted as zero, and a rig where no
        # camera reported at all is not OK: an unmeasured mount must not look like a good one.
        deviations: list[float] = [abs(value) for camera in self._cameras
                                   for value in mount_deviation(camera) if not math.isnan(value)]
        self._settings.mount = bool(deviations) and max(deviations) <= self._settings.tolerance
