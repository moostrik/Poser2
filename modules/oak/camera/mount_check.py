"""Compare how the cameras are mounted against how the preset says they are."""

import math

from .settings import CameraSettings, MountCheckSettings


class MountCheck:
    """Turns the per-camera IMU readings into one line a person will actually read.

    Two deviations, both in degrees, both compared against the same `tolerance`:

    - **tilt** — ``|tilt_measured - tilt|``. `tilt` is the value the warp mesh was baked with, so
      a camera that does not sit at it is being un-tilted by the wrong amount.
    - **roll** — ``|roll_measured|``. There is no configured roll: `equirect_mesh_points` has no
      roll term at all, so the only correct value is zero, and any roll tips the horizon in a way
      the panorama cannot distinguish from a wrong tilt.

    `fov_factory` is deliberately *not* checked. It comes from the unit's calibration, whose
    pinhole model cannot represent a 127 degree lens, so it disagrees by design — alarming on it
    would mean a warning every launch, which is the fastest way to teach someone to ignore
    warnings.
    """

    def __init__(self, cameras: list[CameraSettings], settings: MountCheckSettings) -> None:
        self._cameras: list[CameraSettings] = cameras
        self._settings: MountCheckSettings = settings

    def update(self) -> None:
        """Recompute the status line. Cheap; call it on the render tick."""
        tilt_deviations: list[tuple[int, float]] = []
        roll_deviations: list[tuple[int, float]] = []

        for index, camera in enumerate(self._cameras):
            measured_tilt: float = camera.readings.tilt_measured
            if not math.isnan(measured_tilt):
                tilt_deviations.append((index, abs(measured_tilt - camera.tilt)))
            measured_roll: float = camera.readings.roll_measured
            if not math.isnan(measured_roll):
                roll_deviations.append((index, abs(measured_roll)))

        # Cameras that never reported are left out entirely rather than counted as zero: a board
        # with no IMU must not be able to drag an average down and make a bad rig look fine.
        if not tilt_deviations and not roll_deviations:
            self._settings.tilt_deviation = 0.0
            self._settings.roll_deviation = 0.0
            self._settings.status = 'mount not measured — no IMU reading from any camera'
            return

        mean_tilt: float = self._mean(tilt_deviations)
        mean_roll: float = self._mean(roll_deviations)
        self._settings.tilt_deviation = mean_tilt
        self._settings.roll_deviation = mean_roll

        reported: int = len({index for index, _ in tilt_deviations} |
                            {index for index, _ in roll_deviations})
        summary: str = f'tilt {mean_tilt:.1f}°, roll {mean_roll:.1f}° ({reported} of {len(self._cameras)})'

        tolerance: float = self._settings.tolerance
        worst_axis, worst_camera, worst_value = self._worst(tilt_deviations, roll_deviations)
        if worst_value > tolerance:
            self._settings.status = (f'!! cam {worst_camera} {worst_axis} {worst_value:.1f}° off '
                                     f'(limit {tolerance:.1f}°) — avg {summary}')
        else:
            self._settings.status = f'mount OK — {summary}'

    @staticmethod
    def _mean(deviations: list[tuple[int, float]]) -> float:
        if not deviations:
            return 0.0
        return sum(value for _, value in deviations) / len(deviations)

    @staticmethod
    def _worst(tilt_deviations: list[tuple[int, float]],
               roll_deviations: list[tuple[int, float]]) -> tuple[str, int, float]:
        """The single worst offender across both axes, so the warning names one thing."""
        candidates: list[tuple[float, str, int]] = (
            [(value, 'tilt', index) for index, value in tilt_deviations] +
            [(value, 'roll', index) for index, value in roll_deviations]
        )
        if not candidates:
            return ('tilt', 0, 0.0)
        value, axis, index = max(candidates)
        return (axis, index, value)
