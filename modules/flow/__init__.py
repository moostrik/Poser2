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
from .motion import DepthView, DepthVelocity, DepthVelocitySettings, GradientFlow, GradientFlowSettings, Patch, \
                    LucasKanadeFlow, LucasKanadeFlowSettings, LucasKanadeFlow3D, LucasKanadeFlow3DSettings

# Bridges: the density's color both make, the 3D bridge by name, the 2D bridge as `bridge_2d` (bridge_2d.DensityBridge, ...)
from .density_color import DensityColor
from . import bridge_2d
from .bridge_3d import  Bridge3D, Bridge3DSettings, \
                        Magnitude, VelocityMagnitude, \
                        decay_weight, \
                        VelocityLowpass, VelocityLowpassSettings, gaussian_weights, blur_sigma, blur_dispatches, ZPositionMix, \
                        VelocityBandpass, VelocityBandpassSettings, Change, WeightedSum, step_peak, band_width, \
                        FrameMax, \
                        VelocityPeak, VelocityPeakSettings, speed_percentile

# Visualization
from .visualization import Visualizer, histogram_grid, histogram_volume_grid, Strip, ALL_CHANNELS, summary_grid, Legend, LegendKind, DrawnSize, \
                           VolumeCut, VolumeCube, CubeColor, VolumeGlyphs, glyph_grid, Cut, cut_size, projects_max, Orbit, PITCH_LIMIT, ZOOM_LIMITS, box_size, axis_labels, \
                           ZPositionView

# Shared by the 2D and 3D fluid
from .time_step import TimeStep, TimeStepMode, hold_tick, viscosity_iterations, MAX_STEP, REFERENCE_STEP
from .finite_guard import guard_report

# 2D fluid simulation
from .fluid_2d import Fluid2D, Fluid2DSettings

# 3D fluid simulation
from .fluid_3d import Fluid3D, Fluid3DSettings, CompositeSettings, Advection, Overlap, DensityBlend, drag_share, swirl_noise
