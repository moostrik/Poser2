"""What the 2D and the 3D bridge share: the `Bridge` they run around their kernels, the settings, the math of the
trails, the blur and the band, the merge of a tick's frames and the peak's Python. How and why:
modules/flow/BRIDGE_3D.md and BRIDGE_2D.md."""

from .trails import decay_weight
from .blur import blur_sigma, blur_dispatches, gaussian_weights, FULL_WIDTH_HALF_MAXIMUM
from .band import step_peak, band_width, change_acceleration_scale
from .change import Change
from .frame_max import FrameMax
from .velocity_peak import VelocityPeak, VelocityPeakSettings, SpeedCount, speed_percentile
from .bridge_settings import BridgeSettings, BridgeFiltersSettings, VelocityLowpassSettings, VelocityBandpassSettings, \
                             MixSettings, AmountMixSettings
from .bridge import Bridge, Lowpass, Bandpass, READINGS, BANDPASS_READINGS, PARTS
