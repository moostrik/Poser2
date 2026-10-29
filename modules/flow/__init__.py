"""Flow processing package — canonical public API.

All external code should import flow classes from this package, not from
subpackages (e.g. ``from modules.flow import FluidFlow``, never
``from modules.flow.fluid import FluidFlow``).
"""

# Base classes
from .flow_base import FlowBase
from .flow_util import FlowUtil

# Optical flow
from .optical import OpticalFlow, OpticalFlowSettings

# Bridge layers
from .bridge import     DensityBridge, DensityBridgeSettings, Magnitude, VelocityMagnitude, \
                        SmoothTrail, VelocitySmoothTrail, SmoothTrailSettings, \
                        TemperatureBridge, TemperatureBridgeSettings

# Visualization
from .visualization import VisualizationFieldSettings, Visualizer, VelocityField

# 2D fluid simulation
from .fluid import FluidFlow, FluidFlowSettings, DepthSettings

# 3D fluid simulation
from .fluid3d import FluidFlow3D
