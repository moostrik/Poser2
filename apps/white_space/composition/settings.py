from modules.settings import BaseSettings, Field, Group
from apps.white_space.composition.test_composition import TestCompositionSettings


class CompositionParams(BaseSettings):
    """Live-tweakable void and wave pattern parameters."""

    # Void zones
    void_width:    Field[float] = Field(0.05,  min=0.0, max=1.0,   step=0.01,  description="Void width (normalised)")
    void_edge:     Field[float] = Field(0.01,  min=0.0, max=1.0,   step=0.005, description="Void edge softness")
    use_void:      Field[bool]  = Field(True,                                   description="Enable void zones")

    # Wave pattern
    pattern_width:  Field[float] = Field(0.2,  min=0.0, max=1.0,   step=0.01, description="Pattern width (normalised)")
    pattern_edge:   Field[float] = Field(0.2,  min=0.0, max=1.0,   step=0.01, description="Pattern edge softness")
    line_sharpness: Field[float] = Field(1.5,  min=0.0, max=10.0,  step=0.1,  description="Line sharpness")
    line_speed:     Field[float] = Field(1.5,  min=0.0, max=10.0,  step=0.1,  description="Line speed")
    line_width:     Field[float] = Field(0.1,  min=0.0, max=1.0,   step=0.01, description="Line width (normalised)")
    line_amount:    Field[float] = Field(20.0, min=0.0, max=100.0, step=1.0,  description="Number of lines")


class CompositorSettings(BaseSettings):
    """Settings for the LED composition thread."""

    # Construction / wiring (INIT — requires restart to take effect)
    max_poses:        Field[int]   = Field(3,     min=1,   max=16,    access=Field.INIT, description="Max tracked poses")
    light_rate:       Field[float] = Field(30.0,  min=1,   max=120,   access=Field.INIT, description="Light output frame rate (fps)")
    light_resolution: Field[int]   = Field(300,   min=10,  max=1000,  access=Field.INIT, visible=False, description="LED strip resolution (pixels)")
    fov_degrees:      Field[float] = Field(110.0, min=60.0, max=180.0, step=0.5, access=Field.INIT,
                                          description="Camera horizontal FOV in degrees — must match PanoramicTracker.fov")

    params: Group[CompositionParams]     = Group(CompositionParams)
    test:   Group[TestCompositionSettings] = Group(TestCompositionSettings)
