"""What every camera has in the panel: its rate, its flips and what it reads back; and what a
rig camera has on top: its mount and lens. A backend's settings class extends one of the two with
the fields of its device, so the preset keys stay flat: `cam_0.fov` and `cam_0.readings.roll_offset`
are the same keys for every rig backend."""

from modules.settings import BaseSettings, Field, Group, Widget


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


class CameraReadings(BaseSettings):
    """What every camera reports back: the rate its frames arrive at. Read-only; a backend fills it."""
    video_fps:        Field[float] = Field(0.0, access=Field.READ, description="Video FPS")
    video_min_fps:    Field[float] = Field(0.0, access=Field.READ, description="Lowest momentary video rate: 1 / the worst gap between frames (fps)")
    video_latency:    Field[float] = Field(0.0, access=Field.READ, unit="ms", description="Frame capture to arrival on the PC")


class RigCameraReadings(CameraReadings):
    """What a rig camera reports back about itself, and the one number that corrects it.

    Nothing in the show reads any of this — it exists for whoever is setting the rig up. All of it
    is read-only except `roll_offset`, which lives here rather than with the mount constants
    because it is the thing you reach for while looking at the roll it corrects. A backend fills
    what its device can measure; the rest stays at its default (NaN for the angles).
    """
    tracker_fps:      Field[float] = Field(0.0, access=Field.READ, description="Tracker updates/s")
    tracker_latency:  Field[float] = Field(0.0, access=Field.READ, unit="ms", description="Frame capture to the tracklets' arrival on the PC")
    tracklets:      Field[int]   = Field(0, access=Field.READ, description="Active tracklets")
    depth_fps:        Field[float] = Field(0.0, access=Field.READ, description="Depth FPS")
    depth_latency:    Field[float] = Field(0.0, access=Field.READ, unit="ms", description="Frame capture to the depth's arrival on the PC")
    # How the camera is actually mounted, as the board reports it — NaN until measured, which is
    # the honest default for a board with no IMU.
    tilt_measured:  Field[float] = Field(float('nan'), access=Field.READ, description="Up-tilt (°) from the camera's own IMU — compare with `tilt`")
    roll_measured:  Field[float] = Field(float('nan'), access=Field.READ, description="Roll (°) about the optical axis, `roll_offset` subtracted. Should be 0")
    # No min/max: `Widget.resolve` sends a bounded float to a slider and an unbounded one to a
    # number box, and this is typed in or filled by the zero button, never dragged.
    roll_offset:    Field[float] = Field(0.0, description="What this camera reads when level (°). Corrects the reading, not the image")
    fov_factory:    Field[float] = Field(float('nan'), access=Field.READ, description="Horizontal field (°) this unit's own calibration spans across the frame")
    lens_error:     Field[float] = Field(float('nan'), access=Field.READ, description="Largest bearing error (°) this unit has under the shared lens")


class TiltCorrectionSettings(BaseSettings):
    """Straightening the frames for the camera's tilt: the IMU's tilt plus `offset`. Live, and the
    same for a camera and for a recording of one."""
    enabled: Field[bool]  = Field(False, widget=Widget.switch, description="Straighten color and depth for the camera's tilt")
    offset:  Field[float] = Field(0.0, min=-60.0, max=60.0, step=0.5, widget=Widget.slider, description="Added to the IMU's tilt (°)")


class CameraSettings(BaseSettings):
    """The camera as every app knows it: its rate, its flips and its readings.

    A backend's settings class extends this with what its device has (the sensor controls, the
    device id). A backend whose device takes these live redeclares the field without `Field.INIT`;
    the subclass's field is the one the panel shows.
    """
    fps:            Field[float] = Field(30.0, min=1.0, max=120.0, access=Field.INIT, description="Camera FPS")
    flip_h:         Field[bool]  = Field(False, access=Field.INIT, description="Flip horizontal")
    flip_v:         Field[bool]  = Field(False, access=Field.INIT, description="Flip vertical")

    readings:     Group[CameraReadings]     = Group(CameraReadings)


class RigCameraSettings(CameraSettings):
    """A camera of a tracking rig, as the tracker and the render know it: a `CameraSettings` with
    its mount and lens, and the readings of a camera that runs a detector and measures its mount.

    The panel shows the backend's fields first and these after them.
    """
    # Relayed down from the camera group so the warp can turn a tilt into pixels.
    fov:           Field[float] = Field(127.0, access=Field.INIT,
                                         description="Azimuth span (°) of the delivered frame, quoted for the full sensor width")
    # The lens itself — see the lens geometry notes in `geometry.py`. Shared by an
    # installation's cameras; 0 / 0 / 0 is the old model, the lens taken to be `fov`.
    lens_fov:       Field[float] = Field(0.0, access=Field.INIT, step=0.1,
                                         description="Field (°) the lens spans across the full sensor width; 0 = same as fov")
    lens_center_x:  Field[float] = Field(0.0, access=Field.INIT, step=0.5,
                                         description="Optical center offset from the frame center (px), full sensor mode")
    lens_center_y:  Field[float] = Field(0.0, access=Field.INIT, step=0.5,
                                         description="Optical center offset from the frame center (px), positive = down")

    readings:     Group[RigCameraReadings]  = Group(RigCameraReadings)

    @property
    def expected_tilt(self) -> float:
        """The up-tilt (°) the mount check compares the IMU with: level, unless a backend's mount
        says otherwise."""
        return 0.0
