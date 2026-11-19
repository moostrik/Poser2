"""Flow processing package — canonical public API.

All external code should import flow classes from this package, not from
subpackages (e.g. ``from modules.flow import Fluid2D``, never
``from modules.flow.fluid_2d import Fluid2D``).
"""

# Ports: named textures with their kinds, what the monitors show
from .port import Port, Ports, TextureKind, ColorSpace, HasPorts, OnDemand, LightPlace, prefixed, ports_of

# Base classes
from .flow_base import FlowBase
from .flow_util import FlowUtil

# View: the flow's view of the frame, before the motion
from .view import fixed_window, view_span, depth_step, view_per_depth_pixel, place_points, pyramid_sizes, DepthFill, DepthCrop, DepthUpscale

# Motion: the optical flow
from .motion import DepthView, DepthVelocity, DepthVelocitySettings, Patch, \
                    LucasKanadeFlow, LucasKanadeFlowSettings, LucasKanadeFlow2D, LucasKanadeFlow2DSettings, \
                    LucasKanadeFlow3D, LucasKanadeFlow3DSettings

# The density's color: the `color` stage
from .color import DensityColor, DensityColorSettings, white_balance_matrix

# Bridges: what both share, then the 2D's and the 3D's, each on its own kernels
from .bridge import     Bridge, BridgeSettings, BridgeFiltersSettings, Lowpass, Bandpass, READINGS, \
                        decay_weight, gaussian_weights, blur_sigma, blur_dispatches, step_peak, band_width, change_acceleration_scale, \
                        Change, FrameMax, VelocityPeak, VelocityPeakSettings, SpeedCount, speed_percentile, \
                        VelocityLowpassSettings, VelocityBandpassSettings
from .bridge_2d import  Bridge2D, VelocityLowpass2D, VelocityBandpass2D, SpeedCount2D
from .bridge_3d import  Bridge3D, Magnitude, VelocityMagnitude, VelocityLowpass, VelocityBandpass, ZPositionMix, WeightedSum

# Visualization
from .visualization import Visualizer, histogram_grid, histogram_volume_grid, Strip, ALL_CHANNELS, summary_grid, Legend, LegendKind, DrawnSize, \
                           VolumeCut, VolumeCube, CubeColor, VolumeGlyphs, glyph_grid, Cut, cut_size, projects_max, Orbit, PITCH_LIMIT, ZOOM_LIMITS, box_size, axis_labels, \
                           ZPositionView

# Shared by the 2D and 3D fluid
from .time_step import TimeStep, TimeStepMode, hold_tick, viscosity_iterations, MAX_STEP, REFERENCE_STEP
from .drag import DRAG_FLOOR, drag_share
from .density_blend import DensityBlend
from .finite_guard import guard_report

# 2D fluid simulation
from .fluid_2d import Fluid2D, Fluid2DSettings

# 3D fluid simulation
from .fluid_3d import Fluid3D, Fluid3DSettings, CompositeSettings, Advection, Overlap, swirl_noise
