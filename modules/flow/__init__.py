"""Flow processing package — canonical public API.

All external code should import flow classes from this package, not from
subpackages (e.g. ``from modules.flow import FluidFlow``, never
``from modules.flow.fluid import FluidFlow``).
"""

# Base classes
from .flow_base import FlowBase
from .flow_util import FlowUtil

# Optical flow
from .optical import DepthVelocity, catch_up_share, color_motion, filter_depth_steps, track_depth, track_weights, depth_threshold, \
                     GradientFlow, GradientFlowSettings, LucasKanadeFlow, Patch, LucasKanadeFlowSettings, LucasKanadeSettings, DepthTrackSettings, lucas_kanade, median_displacement, threshold_displacement, filter_displacement, displacement_velocity

# Bridge layers: the 3D bridge by name, the 2D bridge as `bridge_2d` (bridge_2d.DensityBridge, ...)
from .bridge import     bridge_2d, DensityBridge, DensityBridgeSettings, Magnitude, VelocityMagnitude, \
                        BlurPosition, KineticBridge, KineticBridgeSettings, KineticStrip, KINETIC_STRIP_LANES, decay_weight, project, stop_gate, hold_release, drop, fill_reservoir, advect, \
                        TrailBridge, TrailBridgeSettings, shape_velocity, \
                        TemperatureBridge, TemperatureBridgeSettings

# Visualization
from .visualization import VisualizationFieldSettings, Visualizer, VelocityField, SignedField

# Time step shared by the 2D and 3D fluid
from .time_step import TimeStep, TimeStepMode

# 2D fluid simulation
from .fluid import FluidFlow, FluidFlowSettings, DepthSettings

# 3D fluid simulation
from .fluid3d import FluidFlow3D
