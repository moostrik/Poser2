"""What every camera has in the panel: its mount, its rate and what it reads back. A backend's
settings class extends `CameraSettings` with the fields of its device, so the preset keys stay
flat: `cam_0.tilt` and `cam_0.readings.roll_offset` are the same keys for every backend."""

from modules.settings import BaseSettings, Field, Group, Widget


class CameraCheckSettings(BaseSettings):
    """Pinned verdicts saying whether the cameras are mounted and running the way the preset assumes.

    The per-camera readings live on each `CameraSettings`, groups deep in the panel, which is
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


class CameraReadings(BaseSettings):
    """What the camera reports back about itself, and the one number that corrects it.

    Nothing in the show reads any of this — it exists for whoever is setting the rig up. All of it
    is read-only except `roll_offset`, which lives here rather than with the mount constants
    because it is the thing you reach for while looking at the roll it corrects. A backend fills
    what its device can measure; the rest stays at its default (NaN for the angles).
    """
    video_fps:        Field[float] = Field(0.0, access=Field.READ, description="Video FPS")
    video_min_fps:    Field[float] = Field(0.0, access=Field.READ, description="Lowest momentary video rate: 1 / the worst gap between frames (fps)")
    tracker_fps:      Field[float] = Field(0.0, access=Field.READ, description="Tracker updates/s")
    tracklets:      Field[int]   = Field(0, access=Field.READ, description="Active tracklets")
    # How the camera is actually mounted, as the board reports it — NaN until measured, which is
    # the honest default for a board with no IMU.
    tilt_measured:  Field[float] = Field(float('nan'), access=Field.READ, description="Up-tilt (°) from the camera's own IMU — compare with `tilt`")
    roll_measured:  Field[float] = Field(float('nan'), access=Field.READ, description="Roll (°) about the optical axis, `roll_offset` subtracted. Should be 0")
    # No min/max: `Widget.resolve` sends a bounded float to a slider and an unbounded one to a
    # number box, and this is typed in or filled by the zero button, never dragged.
    roll_offset:    Field[float] = Field(0.0, description="What this camera reads when level (°). Corrects the reading, not the image")
    fov_factory:    Field[float] = Field(float('nan'), access=Field.READ, description="Horizontal field (°) this unit's own calibration spans across the frame")
    lens_error:     Field[float] = Field(float('nan'), access=Field.READ, description="Largest bearing error (°) this unit has under the shared lens")


class CameraSettings(BaseSettings):
    """The camera as the apps, the tracker and the render know it: its rate and its mount.

    A backend's settings class extends this with what shapes its own pipeline (the sensor mode,
    the device id, the sensor controls). The panel shows the backend's fields first and these
    after them.
    """
    fps:            Field[float] = Field(30.0, min=1.0, max=120.0, access=Field.INIT, description="Camera FPS")
    flip_h:         Field[bool]  = Field(False, access=Field.INIT, description="Flip horizontal")
    flip_v:         Field[bool]  = Field(False, access=Field.INIT, description="Flip vertical")
    tilt:           Field[float] = Field(0.0, access=Field.INIT,
                                         description="Camera up-tilt (°), positive = aimed up. Exclusive with keystone")
    keystone:       Field[float] = Field(0.0, access=Field.INIT,
                                         description="Keystone (fraction of frame). Exclusive with tilt")
    # Relayed down from the camera group so the warp can turn `tilt` into pixels.
    fov:            Field[float] = Field(127.0, access=Field.INIT,
                                         description="Azimuth span (°) of the delivered frame, quoted for the full sensor width")
    # The lens itself — see the lens geometry notes in `geometry.py`. Shared by an
    # installation's cameras; 0 / 0 / 0 is the old model, the lens taken to be `fov`.
    lens_fov:       Field[float] = Field(0.0, access=Field.INIT, step=0.1,
                                         description="Field (°) the lens spans across the full sensor width; 0 = same as fov")
    lens_centre_x:  Field[float] = Field(0.0, access=Field.INIT, step=0.5,
                                         description="Optical centre offset from the frame centre (px), full sensor mode")
    lens_centre_y:  Field[float] = Field(0.0, access=Field.INIT, step=0.5,
                                         description="Optical centre offset from the frame centre (px), positive = down")

    readings:     Group[CameraReadings]     = Group(CameraReadings)
