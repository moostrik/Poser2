"""Compare what the cameras report about themselves against what the preset says they should be."""

import math
from collections.abc import Sequence

from modules.settings import BaseSettings, Field, Widget

from .settings import RigCameraSettings


class CameraCheckSettings(BaseSettings):
    """Pinned verdicts saying whether the cameras are mounted and running the way the preset assumes.

    The per-camera readings live on each `RigCameraSettings`, groups deep in the panel, which is
    where a readout goes to be ignored. The pinned indicators say good or bad; which camera and
    which value is drawn on that camera's own view in the renderer.
    """
    mount_tolerance: Field[float] = Field(2.0, min=0.5, max=10.0, step=0.5,
                                          description="Tilt or roll deviation (°) a camera may have before the mount warns")
    mount:           Field[bool]  = Field(True, access=Field.READ, pinned=True, widget=Widget.status,
                                          description="Camera tilt and roll within tolerance; each camera view shows its error")
    camera_fps:      Field[bool]  = Field(False, access=Field.READ, pinned=True, widget=Widget.status,
                                          description="Camera frame rates within 5 % of fps; each camera view shows its rate")
    video_rate:      Field[bool]  = Field(False, access=Field.READ, pinned=True, widget=Widget.status, label="Video",
                                          description="Every camera's momentary video rate above half its fps")
    video_fps_avg:   Field[float] = Field(0.0, access=Field.READ, pinned=True, label="Video", unit="fps",
                                          description="Slowest camera's average video rate")
    video_min_fps:   Field[float] = Field(0.0, access=Field.READ, pinned=True, label="Video min", unit="fps",
                                          description="Worst momentary video rate across the cameras")
    video_latency:   Field[float] = Field(0.0, access=Field.READ, pinned=True, label="Video latency", unit="ms",
                                          description="Slowest camera's frame capture to arrival on the PC")
    depth_latency:   Field[float] = Field(0.0, access=Field.READ, pinned=True, label="Depth latency", unit="ms",
                                          description="Slowest camera's depth capture to arrival on the PC")
    link:            Field[str]   = Field("", access=Field.READ, pinned=True, label="Link", description="How the cameras reach the PC")


# The fraction a camera's measured frame rate may drift from its configured `fps` before it warns.
FPS_TOLERANCE: float = 0.05

# The fraction of its configured `fps` a camera's momentary video rate may dip to before it warns
# (delivery stutter: one late frame is a dip in the momentary rate the average hides).
MIN_RATE_FACTOR: float = 0.5


def mount_deviation(camera: RigCameraSettings) -> tuple[float, float]:
    """The camera's signed ``(tilt, roll)`` deviation in degrees; NaN where it has no reading.

    - **tilt** — ``tilt_measured - expected_tilt``. On an OAK-D that is the `tilt` the warp mesh
      was baked with, so a camera that does not sit at it is being un-tilted by the wrong amount.
    - **roll** — ``roll_measured``. There is no configured roll: `warp_mesh_points` has no
      roll term at all, so the only correct value is zero, and any roll tips the horizon in a way
      the panorama cannot distinguish from a wrong tilt.
    """
    return (camera.readings.tilt_measured - camera.expected_tilt, camera.readings.roll_measured)


def fps_deviation(camera: RigCameraSettings) -> float:
    """The camera's signed frame-rate deviation as a fraction of its configured `fps`."""
    return camera.readings.video_fps / camera.fps - 1.0


class CameraCheck:
    """Turns the per-camera readings into two verdicts a person will actually read.

    - **mount** — every tilt and roll deviation (`mount_deviation`) within `mount_tolerance`.
      Cameras without an IMU are left out; they are a property of the board, not something to fix
      on site, so they never warn. The camera view says `no IMU` instead.
    - **camera_fps** — every camera's video frame rate within `FPS_TOLERANCE` of its `fps`
      (`fps_deviation`). A camera that delivers no frames reads 0 and warns.
    - **video_rate** — every camera's momentary video rate (1 / its worst frame gap) above
      `MIN_RATE_FACTOR` of its `fps`: the average rate can be fine while single frames arrive
      late, which is what a viewer sees as stutter. The slowest average and the worst momentary
      rate are published as `video_fps_avg` and `video_min_fps`.

    The slowest camera's `video_latency` and `depth_latency` are published as they are, no verdict:
    0 where a backend does not measure them. So is `link`, each different link once: empty where a
    backend does not report one.

    Which camera and axis is off is not repeated here: the renderer draws each camera's numbers
    on its own view (`CameraReadingsLayer`).

    `fov_factory` and `lens_error` are deliberately *not* checked. They describe the unit's lens
    against the shared one, which is a property of the build, not of how the tripod was set
    this morning — alarming on them would mean a warning every launch, which is the fastest way
    to teach someone to ignore warnings.
    """

    def __init__(self, cameras: Sequence[RigCameraSettings], settings: CameraCheckSettings) -> None:
        self._cameras: Sequence[RigCameraSettings] = cameras
        self._settings: CameraCheckSettings = settings

    def update(self, _now: float) -> None:
        """Recompute both verdicts. Cheap; call it on the render tick (the frame time is not needed: the
        verdicts read the cameras' readouts)."""
        mount_deviations: list[float] = [abs(value) for camera in self._cameras
                                         for value in mount_deviation(camera) if not math.isnan(value)]
        self._settings.mount = all(value <= self._settings.mount_tolerance for value in mount_deviations)
        self._settings.camera_fps = all(abs(fps_deviation(camera)) <= FPS_TOLERANCE for camera in self._cameras)
        minimums: list[float] = [camera.readings.video_min_fps for camera in self._cameras]
        self._settings.video_fps_avg = min((camera.readings.video_fps for camera in self._cameras), default=0.0)
        self._settings.video_min_fps = min(minimums, default=0.0)
        self._settings.video_rate = all(minimum >= MIN_RATE_FACTOR * camera.fps
                                        for camera, minimum in zip(self._cameras, minimums))
        self._settings.video_latency = max((camera.readings.video_latency for camera in self._cameras), default=0.0)
        self._settings.depth_latency = max((camera.readings.depth_latency for camera in self._cameras), default=0.0)
        self._settings.link = ', '.join(sorted({camera.readings.link for camera in self._cameras if camera.readings.link}))
