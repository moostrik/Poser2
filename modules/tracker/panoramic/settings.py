from modules.oak import CameraResolution
from modules.settings import BaseSettings, Field, Group, Widget


class SeamSettings(BaseSettings):
    """When two cameras' views of a seam are one person, where a person may be born, and which view
    speaks for them.

    In real units — degrees of world azimuth, a fraction of a measured height, seconds — so a number
    means the same after a change of `fov`, mount or lens. The derived overlap is published on
    `RigSettings`, beside what it comes from.
    """
    dead_zone: Field[float] = Field(5.0, min=0.0, max=20.0, step=0.5,
                                    description="No new person is born within this many ° of a camera's field edge")
    link_angle: Field[float] = Field(8.0, min=0.0, max=40.0, step=0.5,
                                     description="Two cameras' observations within this many ° of azimuth are one person")
    link_height: Field[float] = Field(0.15, min=0.0, max=1.0, step=0.01,
                                      description="Maximum fraction the two measured heights may differ when linking")
    hysteresis: Field[float] = Field(0.9, min=0.1, max=1.0, step=0.05,
                                     description="Edge-distance ratio another camera must beat to take over; lower is stickier")
    # A primary that loses the person hands over at once, skipping `hysteresis`; without this a
    # one-frame miss would switch cameras twice.
    hold: Field[float] = Field(0.5, min=0.0, max=2.0, step=0.05,
                               description="Seconds after a lost primary is replaced before the better-placed camera may take the person back")


class RigSettings(BaseSettings):
    """The installation in metres: where the lenses are (``camera_``), and where people are tracked
    (``zone_``). Measured and taped, not tuned.

    **Every length is a radius from the fixture axis** — the physical centre the installation is
    taped from — and the same number goes into `Rig`, `projection`, the stitch shader and the
    panorama's ``R`` label, unconverted. (``camera_height`` is a height.)

    What depends on them: the parallax correction (the lenses sit on a ring, not at a shared centre;
    ``camera_radius`` 0 disables it), the parallax depth and overlap band derived from the zone
    (`Rig._update_parallax_depth`), and the far edge past which ``zone_filter`` ignores people.

    The rest is read-only, published by the tracker so the panel and the panorama use the numbers it
    tracks with.
    """
    camera_radius: Field[float] = Field(0.0, min=0.0, max=1.0, step=0.01,
                                        description="Distance (m) of each lens from the fixture axis. 0 disables the parallax correction")
    camera_height: Field[float] = Field(0.5, min=0.1, max=3.0, step=0.01,
                                        description="Lens height above the floor (m), measured")
    zone_min_radius: Field[float] = Field(1.5, min=0.25, max=10.0, step=0.05, newline=True,
                                          description="Radius (m) of the tracked floor's near edge — the nearest distance claimed")
    zone_max_radius: Field[float] = Field(3.5, min=0.5, max=15.0, step=0.05,
                                          description="Radius (m) of its far edge — sets the overlap band, and where zone_filter ignores people")
    parallax_radius: Field[float] = Field(2.1, access=Field.READ,
                                          description="Radius (m) the world azimuth is corrected at — derived, the zone's harmonic mean")
    # In world azimuth, as the panorama's grid measures it; the `Rig` keeps the local-angle
    # equivalent it tests to itself.
    overlap: Field[float] = Field(0.0, access=Field.READ,
                                  description="Azimuth (°) two neighbours share, at the zone's far edge")
    # The delivered frame's shape. The two angles are the whole row model (`projection.row_model`), so
    # nothing in row space is published. `angle_top` is the frame's top row, which may sit above
    # what the sensor fills.
    hfov: Field[float] = Field(127.0, access=Field.READ, newline=True,
                               description="Azimuth span (°) of the delivered frame, left edge to right")
    vfov: Field[float] = Field(79.5, access=Field.READ,
                              description="Angle span (°) of the delivered frame, bottom row to top row")
    tilt: Field[float] = Field(0.0, access=Field.READ,
                               description="Camera up-tilt (°), as the frame was built with")
    angle_bottom: Field[float] = Field(-39.7, access=Field.READ,
                                       description="Angle (°) from eye level of the frame's bottom row; negative is below")
    angle_top: Field[float] = Field(39.7, access=Field.READ,
                                    description="Angle (°) from eye level of the frame's top row")
    # What the frame allows, against what the sensor actually fills (CALIBRATION.md, *Tilt — derived
    # from the build*). Hands twice, because the sensor's top edge falls toward the seams.
    feet_from: Field[float] = Field(0.0, access=Field.READ, newline=True,
                                    description="Nearest radius (m) with the feet in frame")
    hands_from: Field[float] = Field(0.0, access=Field.READ,
                                     description="Nearest radius (m) with 2.2 m raised hands in frame, on a camera's axis")
    hands_seam: Field[float] = Field(0.0, access=Field.READ,
                                     description="Nearest radius (m) with 2.2 m raised hands in frame, on a seam")


class TrackerSettings(BaseSettings):
    fov: Field[float] = Field(110.0, access=Field.INIT)
    # The frame geometry, shared from the camera group: what the warp was built with.
    resolution: Field[CameraResolution] = Field(CameraResolution.P800, access=Field.INIT,
                                                description="Sensor mode, shared — the frame's shape")
    frame_height: Field[int] = Field(0, access=Field.INIT,
                                    description="Delivered frame height (px), shared; 0 = derived from the tilt")
    tilt: Field[float] = Field(0.0, access=Field.INIT, description="Camera up-tilt (°), shared")
    lens_fov: Field[float] = Field(0.0, access=Field.INIT, description="Lens field (°) across the sensor width, shared; 0 = fov")
    lens_centre_x: Field[float] = Field(0.0, access=Field.INIT, description="Optical centre offset (px), shared")
    lens_centre_y: Field[float] = Field(0.0, access=Field.INIT, description="Optical centre offset (px), shared")
    # The intake's filters. A detection one rejects is drawn grey on the panorama, labelled with the
    # filter; a person already tracked goes LOST instead, their own label naming it.
    age_filter: Field[int] = Field(5, min=0, max=9, step=1,
                                   description="Minimum device track age (frames) before a detection counts")
    height_filter: Field[float] = Field(0.25, min=0.0, max=1.0, step=0.05,
                                        description="Minimum box height (fraction of frame) before a detection counts")
    zone_filter: Field[bool] = Field(True, widget=Widget.switch,
                                     description="Ignore people past rig.zone_max_radius")
    # A property of the detector, not the site, hence not in `rig`. Tuned by walking one person out
    # until the panorama's `H` stops drifting.
    foot_offset: Field[float] = Field(0.0, min=0.0, max=0.2, step=0.005,
                                      description="How far below the feet the detector's box bottom sits (frame heights)")
    # The same camera re-finding a person, anywhere in its field: not a seam setting.
    reacquire_angle: Field[float] = Field(5.0, min=0.0, max=20.0, step=0.5,
                                          description="How far (°) a re-acquired person may be from the lost one and still be them")
    # How long they keep a pose from their last box is the pose side's `detection_timeout`.
    lost_timeout: Field[float] = Field(2.0, min=1.0, max=5.0, step=0.1,
                                       description="Seconds a lost person is remembered, for seam links and re-acquisition")
    seam: Group[SeamSettings] = Group(SeamSettings)
    rig: Group[RigSettings] = Group(RigSettings)
