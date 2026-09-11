from modules.oak import CameraResolution
from modules.settings import BaseSettings, Field, Group


class SeamAngles(BaseSettings):
    fov: Field[float] = Field(0.0, access=Field.READ, description="Camera FOV (°)")
    overlap: Field[float] = Field(0.0, access=Field.READ, description="Camera FOV overlap zone (°)")
    reject: Field[float] = Field(0.0, access=Field.READ, description="Dead zone at camera edges (°)")
    reach: Field[float] = Field(0.0, access=Field.READ, description="Cross-camera matching zone from camera edges (°)")


class SeamSettings(BaseSettings):
    reject: Field[float] = Field(0.5, min=0.0, max=0.75, step=0.05,
                                 description="Dead zone size as a fraction of the overlap zone.")
    reach: Field[float] = Field(1.3, min=1.0, max=1.5, step=0.05,
                                description="Matching zone size as a fraction of the overlap zone.")
    hysteresis: Field[float] = Field(0.9, min=0.1, max=1.0, step=0.05,
                                     description="Lower values make active camera stickier.")
    max_height_diff: Field[float] = Field(0.15, min=0.0, max=0.5, step=0.01,
                                          description="Maximum ROI height difference for matching two observations.")
    relink_angle: Field[float] = Field(5.0, min=0.0, max=20.0, step=0.5,
                                       description="How far (°) a new observation may be from a lost one in the SAME camera and still be judged the same person. The device tracker has no appearance model, so a person it drops and re-acquires arrives under a new id; this is what keeps their identity. Small: two people standing closer than this could be confused.")
    angles: Group[SeamAngles] = Group(SeamAngles)


class ParallaxSettings(BaseSettings):
    """
    Corrects for the cameras sitting on a ring rather than at a shared optical
    centre. Each camera is ``ring_radius`` metres from the rig centre, so the
    same person is seen at different world angles by neighbouring cameras — a
    disagreement of several degrees at the seams. Distance to the person is
    estimated from where their feet meet the floor, which needs only the lens
    height above it, and that is enough to re-project each observation to the
    shared centre.

    Both numbers are *measured with a tape*, not tuned. At ``ring_radius = 0``
    the correction is disabled (identity).
    """
    ring_radius: Field[float] = Field(0.0, min=0.0, max=1.0, step=0.01,
                                      description="Camera distance from rig centre (m), measured. 0 disables parallax correction.")
    camera_height: Field[float] = Field(0.5, min=0.1, max=3.0, step=0.01,
                                        description="Lens height above the floor (m), measured. The only constant the distance estimate needs — nothing about the person enters it.")
    vfov: Field[float] = Field(79.5, access=Field.READ,
                              description="Vertical field (°) of the detection frame, for the distance estimate. Derived from the camera's horizontal FOV and the frame's shape, so it follows every resolution and crop change on its own.")


class TrackerSettings(BaseSettings):
    fov: Field[float] = Field(110.0, access=Field.INIT)
    resolution: Field[CameraResolution] = Field(CameraResolution.P800, access=Field.INIT,
                                                description="Sensor mode, shared from the camera group. Only the frame's shape is used, to derive the vertical field.")
    min_age: Field[int] = Field(5, min=0, max=9, step=1,
                                description="Minimum age in frames before a tracklet is considered.")
    min_height: Field[float] = Field(0.25, min=0.0, max=1.0, step=0.05,
                                     description="Minimum ROI height to accept a tracklet.")
    timeout: Field[float] = Field(2.0, min=1.0, max=5.0, step=0.1,
                                  description="Seconds a lost observation keeps anchoring before it is retired. It must outlast a seam crossing: the far camera has to pick a person up before the near one's observation is gone.")
    emit_hold: Field[float] = Field(0.3, min=0.0, max=2.0, step=0.05,
                                    description="Seconds a world keeps being emitted after its last detection. Shorter than `timeout` on purpose: a person who walks out should stop driving the light and the sound long before their observation stops anchoring.")
    seam: Group[SeamSettings] = Group(SeamSettings)
    parallax: Group[ParallaxSettings] = Group(ParallaxSettings)
