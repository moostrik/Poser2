"""Flow processing package — canonical public API.

All external code should import flow classes from this package, not from
subpackages (e.g. ``from modules.flow import Fluid2D``, never
``from modules.flow.fluid_2d import Fluid2D``).
"""

# Ports: named textures with their kinds, what the monitors show
from .port import Port, Ports, TextureKind, HasPorts, OnDemand, LightPlace, prefixed, ports_of

# Base classes
from .flow_base import FlowBase
from .flow_util import FlowUtil

# Motion: the optical flow
from .motion import DepthView, DepthVelocity, DepthVelocitySettings, catch_up_share, color_motion, filter_depth_steps, track_depth, track_weights, depth_threshold, \
                    GradientFlow, GradientFlowSettings, Patch, LucasKanadeFlow, LucasKanadeFlowSettings, lucas_kanade, threshold_displacement, \
                    LucasKanadeFlow3D, LucasKanadeFlow3DSettings, filter_displacement, displacement_velocity

# Bridges: the 3D bridge by name, the 2D bridge as `bridge_2d` (bridge_2d.DensityBridge, ...)
from . import bridge_2d
from .bridge_3d import  DensityBridge, DensityBridgeSettings, Magnitude, VelocityMagnitude, \
                        decay_weight, direction, advect, \
                        VelocityLowpass, VelocityLowpassSettings, velocity_trail, trail_acceleration, \
                        VelocityBandpass, VelocityBandpassSettings, Change, WeightedSum, velocity_change, step_peak, moving_mean, start_region_share, stop_region_share, change_velocity, band_width, change_acceleration_scale, whole_change, \
                        TemperatureBridge, TemperatureBridgeSettings

# Visualization
from .visualization import Visualizer, histogram_grid, histogram_volume_grid, Strip, ALL_CHANNELS, summary_grid, Legend, LegendKind, DrawnSize, \
                           VolumeCut, VolumeCube, CubeColor, VolumeGlyphs, glyph_grid, Cut, cut_size, projects_max, Orbit, PITCH_LIMIT, ZOOM_LIMITS, box_size, axis_labels

# Shared by the 2D and 3D fluid
from .time_step import TimeStep, TimeStepMode, hold_tick, viscosity_iterations, MAX_STEP, REFERENCE_STEP
from .wall_mode import WallMode
from .finite_guard import FiniteGuard, GuardSlot, guard_report

# 2D fluid simulation
from .fluid_2d import Fluid2D, Fluid2DSettings

# 3D fluid simulation
from .fluid_3d import Fluid3D, Fluid3DSettings, CompositeSettings, Advection, ParticleDraw, Overlap, DRAG_FLOOR, drag_share, drag_gate, drag_velocity_step, swirl_noise
