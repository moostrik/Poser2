"""Flow processing package — canonical public API.

All external code should import flow classes from this package, not from
subpackages (e.g. ``from modules.flow import FluidFlow``, never
``from modules.flow.fluid import FluidFlow``).
"""

# Base classes
from .flow_base import FlowBase
from .flow_util import FlowUtil

# Optical flow
from .optical import GradientFlow, GradientFlowSettings, LucasKanadeFlow, LucasKanadeFlowSettings, lucas_kanade, threshold_displacement, displacement_velocity

# Bridge layers
from .bridge import     DensityBridge, DensityBridgeSettings, Magnitude, VelocityMagnitude, \
                        BlurPosition, KineticBridge, KineticBridgeSettings, KineticStrip, KINETIC_STRIP_LANES, decay_weight, project, stop_gate, hold_release, drop, fill_reservoir, smooth_depth, depth_speed, warped, depth_threshold, advect, \
                        DepthVelocity, DepthVelocitySettings, \
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
