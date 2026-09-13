"""Compare what the cameras report about themselves against what the preset says they should be."""

import math

from .settings import CameraSettings, CameraCheckSettings

# The fraction a camera's measured frame rate may drift from its configured `fps` before it warns.
FPS_TOLERANCE: float = 0.05


def mount_deviation(camera: CameraSettings) -> tuple[float, float]:
    """The camera's signed ``(tilt, roll)`` deviation in degrees; NaN where it has no reading.

    - **tilt** — ``tilt_measured - tilt``. `tilt` is the value the warp mesh was baked with, so
      a camera that does not sit at it is being un-tilted by the wrong amount.
    - **roll** — ``roll_measured``. There is no configured roll: `warp_mesh_points` has no
      roll term at all, so the only correct value is zero, and any roll tips the horizon in a way
      the panorama cannot distinguish from a wrong tilt.
    """
    return (camera.readings.tilt_measured - camera.tilt, camera.readings.roll_measured)


def fps_deviation(camera: CameraSettings) -> float:
    """The camera's signed frame-rate deviation as a fraction of its configured `fps`."""
    return camera.readings.video_fps / camera.fps - 1.0


class CameraCheck:
    """Turns the per-camera readings into two verdicts a person will actually read.

    - **mount** — every tilt and roll deviation (`mount_deviation`) within `mount_tolerance`.
      Cameras without an IMU are left out; they are a property of the board, not something to fix
      on site, so they never warn. The camera view says `no IMU` instead.
    - **camera_fps** — every camera's video frame rate within `FPS_TOLERANCE` of its `fps`
      (`fps_deviation`). A camera that delivers no frames reads 0 and warns.

    Which camera and axis is off is not repeated here: the renderer draws each camera's numbers
    on its own view (`CameraReadingsLayer`).

    `fov_factory` and `lens_error` are deliberately *not* checked. They describe the unit's lens
    against the shared one, which is a property of the build, not of how the tripod was set
    this morning — alarming on them would mean a warning every launch, which is the fastest way
    to teach someone to ignore warnings.
    """

    def __init__(self, cameras: list[CameraSettings], settings: CameraCheckSettings) -> None:
        self._cameras: list[CameraSettings] = cameras
        self._settings: CameraCheckSettings = settings

    def update(self) -> None:
        """Recompute both verdicts. Cheap; call it on the render tick."""
        mount_deviations: list[float] = [abs(value) for camera in self._cameras
                                         for value in mount_deviation(camera) if not math.isnan(value)]
        self._settings.mount = all(value <= self._settings.mount_tolerance for value in mount_deviations)
        self._settings.camera_fps = all(abs(fps_deviation(camera)) <= FPS_TOLERANCE for camera in self._cameras)
