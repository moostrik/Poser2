"""Settings of the pipeline from camera image to grin: one card per stage, the face crop (SMILE's
own) and then the stages of the ComfyUI workflow.

The panel is the source of truth: every run fills these values into the workflow, whose own values
only served as the defaults here.
"""

from enum import Enum

from modules.face.settings import FaceSettings
from modules.settings import BaseSettings, Field, Group, Widget


class ConnectionSettings(BaseSettings):
    host:          Field[str]   = Field("127.0.0.1", access=Field.INIT, widget=Widget.ip_field, description="ComfyUI server address")
    port:          Field[int]   = Field(8188, access=Field.INIT, widget=Widget.number_field, description="ComfyUI server port")
    workflow:      Field[str]   = Field("data/comfy/AIIAv3 portrait grin v3.json", access=Field.INIT, description="Workflow in ComfyUI API format (structure only)")
    job_timeout:   Field[float] = Field(90.0, min=10.0, max=300.0, step=5.0, description="Give up on a generation after this many seconds")

    connected:     Field[bool]  = Field(False, access=Field.READ, description="Connected to ComfyUI")
    status:        Field[str]   = Field("", access=Field.READ, description="Generator status")
    last_duration: Field[float] = Field(0.0, access=Field.READ, pinned=True, description="Duration of the last generation (s)")


class DescribeSettings(BaseSettings):
    instruction:  Field[str] = Field("Describe this person for a text-to-image prompt. Only visible physical features: gender, approximate age, race and ethnicity, hair style, face shape, facial features. No clothing, no background, no expression, no mention of color, no negatives. Short comma-separated phrases on one line. Output only the phrases. ",
                                     widget=Widget.textarea, description="What QwenVL is asked about the face")
    max_tokens:   Field[int] = Field(120, min=64, max=512, step=8, description="Maximum length of the description (tokens)")
    prefix:       Field[str] = Field("", description="Text before the description")
    suffix:       Field[str] = Field(" (super model: 1.5), (serious expression: 2.2)", description="Text after the description")
    description:  Field[str] = Field("", access=Field.READ, widget=Widget.textarea, description="Last prompt: prefix + description + suffix")


class StyleSettings(BaseSettings):
    repeat:     Field[int] = Field(26, min=1, max=100, step=1, description="Visitors per style before switching to the next")
    current:    Field[str] = Field("A", access=Field.READ, description="Style of the current visitor: A or B")
    style_a:    Field[str] = Field("BREAK nude portrait by (David LaChapelle: 1.4), bold artificial hues, high contrast, harsh dramatic light, 2000s BREAK",
                                   widget=Widget.textarea, description="Style of movement 0000 (music: Smooth)")
    negative_a: Field[str] = Field("(text, logo), clothing, (necklace), blurred, illustration, 3d, 2d, painting, cartoons, crooked teeth, ugly teeth",
                                   widget=Widget.textarea, description="Negative prompt of style A")
    style_b:    Field[str] = Field("portrait by (Mert and Marcus: 2.0), high gloss, dramatic chiaroscuro, studio lighting, (bright color solid background: 2.0)",
                                   widget=Widget.textarea, description="Style of movement 0001 (music: Party)")
    negative_b: Field[str] = Field("(text, logo), clothing, (necklace), blurred, illustration, 3d, 2d, painting, cartoons, crooked teeth, ugly teeth",
                                   widget=Widget.textarea, description="Negative prompt of style B")


class PortraitSettings(BaseSettings):
    sampler_options:   Field[list[str]] = Field(["dpmpp_sde"], access=Field.READ, visible=False, description="Samplers offered by ComfyUI")
    scheduler_options: Field[list[str]] = Field(["exponential"], access=Field.READ, visible=False, description="Schedulers offered by ComfyUI")

    steps:             Field[int]   = Field(6, min=1, max=30, step=1, description="Sampler steps")
    cfg:                Field[float] = Field(1.5, min=0.5, max=10.0, step=0.1, description="Prompt guidance (CFG)")
    sampler:            Field[str]   = Field("dpmpp_sde", widget=Widget.text_select, options=sampler_options, description="Sampler")
    scheduler:          Field[str]   = Field("exponential", widget=Widget.text_select, options=scheduler_options, description="Scheduler")
    embedding_strength: Field[float] = Field(1.0, min=0.0, max=2.0, step=0.05, newline=True, description="InstantID face likeness (ip weight)")
    keypoint_strength:  Field[float] = Field(1.0, min=0.0, max=2.0, step=0.05, description="InstantID face pose (keypoints ControlNet)")
    depth_strength:     Field[float] = Field(0.5, min=0.0, max=2.0, step=0.05, newline=True, description="Depth ControlNet strength")
    depth_end:          Field[float] = Field(0.5, min=0.0, max=1.0, step=0.05, description="Depth ControlNet stops at this part of the steps")
    full_preview:       Field[bool]  = Field(True, newline=True, description="Step previews decoded with the real VAE (off: ComfyUI's quick previews)")
    fixed_seed:         Field[bool]  = Field(False, description="Same seeds every run, to compare settings (off: random)")


class LoraSettings(BaseSettings):
    clothing: Field[float] = Field(1.0, min=-5.0, max=5.0, step=0.05, description="Clothing slider LoRA strength (0 = off)")
    looking:  Field[float] = Field(1.0, min=-5.0, max=5.0, step=0.05, description="Looking-at-viewer LoRA strength (0 = off)")
    dark:     Field[float] = Field(-0.5, min=-5.0, max=5.0, step=0.05, description="Dark LoRA strength (0 = off)")


class Ease(Enum):
    SMOOTHSTEP  = 'smoothstep'
    LINEAR      = 'linear'
    EASE_IN     = 'ease_in'
    EASE_OUT    = 'ease_out'
    SINE_IN_OUT = 'sine_in_out'


class Prefer(Enum):
    EVEN        = 'even'
    MIN         = 'min'
    MIN_STRONG  = 'min_strong'
    MAX         = 'max'
    MAX_STRONG  = 'max_strong'
    ENDS        = 'ends'
    ENDS_STRONG = 'ends_strong'


class GrinSettings(BaseSettings):
    loop:              Field[bool]  = Field(False, description="Keep replaying the grin with the current values, without a new portrait (testing)")
    still:              Field[bool]  = Field(False, description="Show only the peak pose as a still that follows the settings (testing)")
    still_position:     Field[float] = Field(1.0, min=0.0, max=1.0, step=0.01, description="Where the still sits in every range: 0 = min, 1 = max")
    frames:             Field[int]   = Field(150, min=10, max=300, step=10, newline=True, description="Frames of the grin; it lasts frames / Grin fps (Sequence tab)")
    face_region:        Field[float] = Field(2.0, min=1.5, max=3.0, step=0.1, description="Area around the face that LivePortrait animates, relative to the face")
    stitching:          Field[bool]  = Field(True, description="LivePortrait stitching (shifts the head; off keeps it in place)")
    ease:              Field[Ease]  = Field(Ease.SINE_IN_OUT, newline=True, description="Curve from neutral to the peak and back")
    midpoint:           Field[float] = Field(0.7, min=0.0, max=1.0, step=0.01, description="Moment of the peak, as part of the animation")
    midpoint_deviation: Field[float] = Field(0.2, min=0.0, max=0.5, step=0.01, description="Each parameter peaks at a random moment within midpoint +- this")

    # every value: one row, the peak is random between min and max (equal = no variation); the still
    # shows min at position 0 and max at 1, as set, so max can be the lower number
    # prefer: where the random value tends to land (a curve over the draw, see workflow.PREFER_CURVES)
    smile_min:   Field[float] = Field(2.0, min=-30.0, max=30.0, step=0.1, newline=True, row_label="Smile", label="Min", description="Smile at the peak: random between min and max")
    smile_max:   Field[float] = Field(2.0, min=-30.0, max=30.0, step=0.1, label="Max", description="Smile at the peak: random between min and max")
    smile_prefer:   Field[Prefer] = Field(Prefer.EVEN, label="Prefer", width=32, description="Where the random value tends to land between min and max")
    aaa_min:     Field[float] = Field(-20.0, min=-30.0, max=30.0, step=0.1, newline=True, row_label="Aaa", label="Min", description="Mouth 'aaa' at the peak: random between min and max")
    aaa_max:     Field[float] = Field(-20.0, min=-30.0, max=30.0, step=0.1, label="Max", description="Mouth 'aaa' at the peak: random between min and max")
    aaa_prefer:     Field[Prefer] = Field(Prefer.EVEN, label="Prefer", width=32, description="Where the random value tends to land between min and max")
    eee_min:     Field[float] = Field(-10.0, min=-30.0, max=30.0, step=0.1, newline=True, row_label="Eee", label="Min", description="Mouth 'eee' at the peak: random between min and max")
    eee_max:     Field[float] = Field(-10.0, min=-30.0, max=30.0, step=0.1, label="Max", description="Mouth 'eee' at the peak: random between min and max")
    eee_prefer:     Field[Prefer] = Field(Prefer.EVEN, label="Prefer", width=32, description="Where the random value tends to land between min and max")
    woo_min:     Field[float] = Field(-20.0, min=-30.0, max=30.0, step=0.1, newline=True, row_label="Woo", label="Min", description="Mouth 'woo' at the peak: random between min and max")
    woo_max:     Field[float] = Field(-20.0, min=-30.0, max=30.0, step=0.1, label="Max", description="Mouth 'woo' at the peak: random between min and max")
    woo_prefer:     Field[Prefer] = Field(Prefer.EVEN, label="Prefer", width=32, description="Where the random value tends to land between min and max")
    eyebrow_min: Field[float] = Field(7.5, min=-30.0, max=30.0, step=0.1, newline=True, row_label="Eyebrow", label="Min", description="Eyebrows at the peak: random between min and max")
    eyebrow_max: Field[float] = Field(7.5, min=-30.0, max=30.0, step=0.1, label="Max", description="Eyebrows at the peak: random between min and max")
    eyebrow_prefer: Field[Prefer] = Field(Prefer.EVEN, label="Prefer", width=32, description="Where the random value tends to land between min and max")
    blink_min:   Field[float] = Field(-9.0, min=-30.0, max=30.0, step=0.1, newline=True, row_label="Blink", label="Min", description="Eyes open (-) / closed (+) at the peak: random between min and max")
    blink_max:   Field[float] = Field(-9.0, min=-30.0, max=30.0, step=0.1, label="Max", description="Eyes open (-) / closed (+) at the peak: random between min and max")
    blink_prefer:   Field[Prefer] = Field(Prefer.EVEN, label="Prefer", width=32, description="Where the random value tends to land between min and max")
    wink_min:    Field[float] = Field(0.0, min=-30.0, max=30.0, step=0.1, newline=True, row_label="Wink", label="Min", description="Wink at the peak: random between min and max")
    wink_max:    Field[float] = Field(0.0, min=-30.0, max=30.0, step=0.1, label="Max", description="Wink at the peak: random between min and max")
    wink_prefer:    Field[Prefer] = Field(Prefer.EVEN, label="Prefer", width=32, description="Where the random value tends to land between min and max")
    pupil_x_min: Field[float] = Field(0.0, min=-15.0, max=15.0, step=0.1, newline=True, row_label="Pupil X", label="Min", description="Gaze left / right at the peak: random between min and max")
    pupil_x_max: Field[float] = Field(0.0, min=-15.0, max=15.0, step=0.1, label="Max", description="Gaze left / right at the peak: random between min and max")
    pupil_x_prefer: Field[Prefer] = Field(Prefer.EVEN, label="Prefer", width=32, description="Where the random value tends to land between min and max")
    pupil_y_min: Field[float] = Field(0.0, min=-15.0, max=15.0, step=0.1, newline=True, row_label="Pupil Y", label="Min", description="Gaze up / down at the peak: random between min and max")
    pupil_y_max: Field[float] = Field(0.0, min=-15.0, max=15.0, step=0.1, label="Max", description="Gaze up / down at the peak: random between min and max")
    pupil_y_prefer: Field[Prefer] = Field(Prefer.EVEN, label="Prefer", width=32, description="Where the random value tends to land between min and max")
    pitch_min:   Field[float] = Field(-15.0, min=-30.0, max=30.0, step=0.5, newline=True, row_label="Pitch", label="Min", description="Head nod (degrees): random between min and max")
    pitch_max:   Field[float] = Field(0.0, min=-30.0, max=30.0, step=0.5, label="Max", description="Head nod (degrees): random between min and max")
    pitch_prefer:   Field[Prefer] = Field(Prefer.EVEN, label="Prefer", width=32, description="Where the random value tends to land between min and max")
    yaw_min:     Field[float] = Field(-15.0, min=-30.0, max=30.0, step=0.5, newline=True, row_label="Yaw", label="Min", description="Head turn (degrees): random between min and max")
    yaw_max:     Field[float] = Field(15.0, min=-30.0, max=30.0, step=0.5, label="Max", description="Head turn (degrees): random between min and max")
    yaw_prefer:     Field[Prefer] = Field(Prefer.EVEN, label="Prefer", width=32, description="Where the random value tends to land between min and max")
    roll_min:    Field[float] = Field(-15.0, min=-30.0, max=30.0, step=0.5, newline=True, row_label="Roll", label="Min", description="Head tilt (degrees): random between min and max")
    roll_max:    Field[float] = Field(15.0, min=-30.0, max=30.0, step=0.5, label="Max", description="Head tilt (degrees): random between min and max")
    roll_prefer:    Field[Prefer] = Field(Prefer.EVEN, label="Prefer", width=32, description="Where the random value tends to land between min and max")


class PipelineSettings(BaseSettings):
    connection: Group[ConnectionSettings] = Group(ConnectionSettings)
    face:       Group[FaceSettings]       = Group(FaceSettings)     # SMILE's own face crop, the image ComfyUI starts from
    describe:  Group[DescribeSettings]   = Group(DescribeSettings)
    style:      Group[StyleSettings]      = Group(StyleSettings)
    portrait:   Group[PortraitSettings]   = Group(PortraitSettings)
    lora:       Group[LoraSettings]       = Group(LoraSettings)
    grin:       Group[GrinSettings]       = Group(GrinSettings)
