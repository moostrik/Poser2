from modules.settings import BaseSettings, Field, Group
from .geometry import DistortAlgorithm


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
    angles: Group[SeamAngles] = Group(SeamAngles)


class TanhSettings(BaseSettings):
    """
    S-curve (sigmoid) undistortion via tanh.

    Maps the normalised x position through:
        x' = 0.5 * (1 + tanh(slope * (2x-1) + cubic * (2x-1)³))

    Use this to correct barrel/pincushion distortion where the deviation
    follows a smooth symmetric S-shape. ``slope`` controls how steeply the
    correction ramps at the centre; ``cubic`` adds an asymmetric higher-order
    bend. Start by tuning ``slope`` alone (typical range 1–3) and only add
    ``cubic`` if the residual error is asymmetric across the frame.

    At slope=0 and cubic=0 the output equals 0.5 for all inputs — this is
    NOT an identity. Set algorithm to NONE when no correction is needed.
    """
    slope: Field[float] = Field(0.0, min=0.0, max=5.0, step=0.1,
                                description="Sigmoid slope. Higher values sharpen the S-curve.")
    cubic: Field[float] = Field(0.0, min=-2.0, max=2.0, step=0.05,
                                description="Cubic modifier added to the sigmoid input.")


class PolySettings(BaseSettings):
    D = """
    Polynomial undistortion.

    Maps the normalised x position through:
        x' = x + k1*(x-0.5) + k2*(x-0.5)³

    Use this for classic lens radial distortion where displacement from
    centre grows roughly linearly (k1) with a cubic roll-off (k2). Positive
    k1 stretches the edges outward; negative pulls them inward. k2 refines
    the correction at the extremes without affecting the centre.

    At k1=0 and k2=0 the transform is an exact identity, so this mode is
    safe to leave active during tuning.
    """
    k1: Field[float] = Field(0.0, min=-0.5, max=0.5, step=0.01,
                             description="Linear coefficient. Positive stretches edges outward, negative pulls inward.")
    k2: Field[float] = Field(0.0, min=-2.0, max=2.0, step=0.05,
                             description="Cubic coefficient. Refines correction at the frame extremes.")


class DistortionSettings(BaseSettings):
    algorithm: Field[DistortAlgorithm] = Field(DistortAlgorithm.NONE)
    poly: Group[PolySettings] = Group(PolySettings)
    tanh: Group[TanhSettings] = Group(TanhSettings)


class TrackerSettings(BaseSettings):
    fov: Field[float] = Field(110.0, min=90.0, max=130.0, step=0.5, visible=False)
    min_age: Field[int] = Field(5, min=0, max=9, step=1,
                                description="Minimum age in frames before a tracklet is considered.")
    min_height: Field[float] = Field(0.25, min=0.0, max=1.0, step=0.05,
                                     description="Minimum ROI height to accept a tracklet.")
    timeout: Field[float] = Field(2.0, min=1.0, max=5.0, step=0.1,
                                  description="Seconds before an inactive tracklet is retired.")
    seam: Group[SeamSettings] = Group(SeamSettings)
    distortion: Group[DistortionSettings] = Group(DistortionSettings)
