"""Flow processing package — canonical public API.

All external code should import flow classes from this package, not from
subpackages (e.g. ``from modules.flow import FluidFlow``, never
``from modules.flow.fluid import FluidFlow``).
"""

# Base classes
from .FlowBase import FlowBase
from .FlowUtil import FlowUtil

# Optical flow
from .optical import OpticalFlow, OpticalFlowConfig

# Bridge layers
from .bridge import (
    DensityBridge, DensityBridgeConfig,
    Magnitude, VelocityMagnitude,
    SmoothTrail, VelocitySmoothTrail, SmoothTrailConfig,
    TemperatureBridge, TemperatureBridgeConfig,
)

# Visualization
from .visualization import VisualisationFieldConfig, Visualizer, VelocityField

# 2D fluid simulation
from .fluid import FluidFlow, FluidFlowConfig

# 3D fluid simulation
from .fluid3d import FluidFlow3D, FluidFlow3DConfig
