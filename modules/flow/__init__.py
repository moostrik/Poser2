"""Flow processing package — canonical public API.

All external code should import flow classes from this package, not from
subpackages (e.g. ``from modules.flow import Fluid2D``, never
``from modules.flow.fluid_2d import Fluid2D``).
"""

# Base classes
from .flow_base import FlowBase
from .flow_util import FlowUtil

# Motion: the optical flow
from .motion import DepthView, DepthVelocity, catch_up_share, color_motion, filter_depth_steps, track_depth, track_weights, depth_threshold, \
                    GradientFlow, GradientFlowSettings, LucasKanadeFlow, Patch, LucasKanadeFlowSettings, LucasKanadeSettings, DepthTrackSettings, lucas_kanade,threshold_displacement, filter_displacement, displacement_velocity

# Bridges: the 3D bridge by name, the 2D bridge as `bridge_2d` (bridge_2d.DensityBridge, ...)
from . import bridge_2d
from .bridge_3d import  DensityBridge, DensityBridgeSettings, Magnitude, VelocityMagnitude, \
                        BlurPosition, KineticBridge, KineticBridgeSettings, KineticStrip, KINETIC_STRIP_LANES, decay_weight, project, stop_gate, hold_release, drop, fill_reservoir, advect, \
                        VelocityTrailBridge, VelocityTrailBridgeSettings, velocity_trail, \
                        VelocityEventBridge, VelocityEventBridgeSettings, velocity_event, event_velocity, \
                        TemperatureBridge, TemperatureBridgeSettings

# Visualization
from .visualization import VisualizationFieldSettings, Visualizer, VelocityField, SignedField

# Shared by the 2D and 3D fluid
from .time_step import TimeStep, TimeStepMode
from .boundary_mode import BoundaryMode

# 2D fluid simulation
from .fluid_2d import Fluid2D, Fluid2DSettings

# 3D fluid simulation
from .fluid_3d import Fluid3D, Fluid3DSettings, VolumeSettings
